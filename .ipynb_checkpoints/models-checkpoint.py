import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        mid_channels = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = x + residual
        return self.relu(x)

class HourglassBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        
        # Сжатие картинки
        self.down1 = ResidualBlock(in_channels, channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResidualBlock(channels, channels)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ResidualBlock(channels, channels)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = ResidualBlock(channels, channels)
        
        # середина модели с неизменной размерностью
        self.center = nn.Sequential(
            ResidualBlock(channels, channels),
            ResidualBlock(channels, channels),
            ResidualBlock(channels, channels)
        )
        
        # Возвращение изначальных размеров
        self.up1 = ResidualBlock(channels, channels)
        self.up2 = ResidualBlock(channels, channels)
        self.up3 = ResidualBlock(channels, channels)
        
        # Прокинутые неизменные слои
        self.upsample1 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        
        # Батч-нормы после апсемплинга
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # сжатие
        d1 = self.down1(x)  # 128×128
        p1 = self.pool1(d1)  # 64×64
        
        d2 = self.down2(p1)  # 64×64
        p2 = self.pool2(d2)  # 32×32
        
        d3 = self.down3(p2)  # 32×32
        p3 = self.pool3(d3)  # 16×16
        
        d4 = self.down4(p3)  # 16×16
        
        # Center
        x = self.center(d4)  # 16×16
        
        # увеличение разрешения с skip connection
        x = self.up1(x)  # 16×16
        x = self.upsample1(x)  # 32×32
        x = self.bn1(x + d3)  # Skip connection
        x = self.relu(x)
        
        x = self.up2(x)  # 32×32
        x = self.upsample2(x)  # 64×64
        x = self.bn2(x + d2)  # Skip connection
        x = self.relu(x)
        
        x = self.up3(x)  # 64×64
        x = self.upsample3(x)  # 128×128
        x = self.bn3(x + d1)  # Skip connection
        x = self.relu(x)
        
        return x

class StackedHourglassNetwork(nn.Module):
    def __init__(self, num_stacks=2, num_keypoints=5, upsample_outputs=False):
        super().__init__()

        self.apply(init_weights)
        self.num_stacks = num_stacks
        self.num_keypoints = num_keypoints
        self.upsample_outputs = upsample_outputs

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256)
        )
        
        # Стек hourglass блоков
        self.hourglasses = nn.ModuleList()
        self.output_blocks = nn.ModuleList()  # Блоки для получения heatmaps
        self.merge_blocks = nn.ModuleList()   # Блоки для объединения с next stack
        
        for i in range(num_stacks):
            # Hourglass блок
            if i == 0:
                self.hourglasses.append(HourglassBlock(256, 256))
            else:
                self.hourglasses.append(HourglassBlock(256 + num_keypoints, 256))
            
            self.output_blocks.append(nn.Sequential(
                ResidualBlock(256, 256),
                nn.Conv2d(256, num_keypoints, kernel_size=1)
            ))
            
            # Блок для подготовки к следующему стеку (если не последний)
            if i < num_stacks - 1:
                self.merge_blocks.append(nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                ))

    def forward(self, x):
        original_size = x.shape[2:]
    
        x = self.initial(x)
    
        outputs = []
        low_res_outputs = []
    
        for i in range(self.num_stacks):
            hourglass_output = self.hourglasses[i](x)
    
            low_res_heatmaps = self.output_blocks[i](hourglass_output)
            low_res_heatmaps = torch.sigmoid(low_res_heatmaps)
            low_res_outputs.append(low_res_heatmaps)
    
            if self.upsample_outputs:
                heatmaps = F.interpolate(
                    low_res_heatmaps,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )
                outputs.append(heatmaps)
            else:
                outputs.append(low_res_heatmaps)
    
            if i < self.num_stacks - 1:
                features = self.merge_blocks[i](hourglass_output)
                x = torch.cat([features, low_res_heatmaps], dim=1)
    
        return outputs

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5, eps=1e-7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = float(s)
        self.m = float(m)
        self.eps = eps

        # Веса классификатора (центры классов)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        #Используем равномерное распределение, диапазон которого зависит от числа входящих и выходящих нейронов
        nn.init.xavier_uniform_(self.weight)

        # Вычисляем тригонометрические константы
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

    def set_margin(self, m: float):
        #Изменяет коэффициент m и пересчитывает тригонометрические константы
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

    def forward(self, embeddings, labels=None):
        # Нормализуем эмбеддинги и веса
        x = F.normalize(embeddings, p=2, dim=1)  # (B, D) - размер батча и размерность эмбеддинга, p=2 -> L2 нормализация
        W = F.normalize(self.weight, p=2, dim=1)  # (N, D) - количество классов и размерность эмбеддинга
        
        # Вычисляем косинусы углов между эмбеддингами и центрами классов, clamp ограничивает снизу нулём, чтобы вдруг не произошло округления к отрицательному числу
        cosine = F.linear(x, W)  # (B, N)
        cosine = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)
        
        # Если меток нет (инференс), просто возвращает масштабированные косинусы
        if labels is None:
            return self.s * cosine
        
        # Для обучения вычисляем sin(theta)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0) + self.eps)
        
        # Вычисляем cos(theta + m) используя тригонометрическую формулу косинуса суммы
        cos_theta_m = cosine * self.cos_m - sine * self.sin_m
        
        # Создаем one-hot вектор для целевых классов
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        
        # Формируем финальные логиты:
        # Для целевого класса используем cos(theta + m)
        # Для остальных классов используем cos(theta)
        logits = cosine.clone()
        logits = logits * (1.0 - one_hot) + cos_theta_m * one_hot
        
        # Масштабируем логиты
        logits = logits * self.s
        
        return logits

class FaceModel(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_size=512,
        use_arcface=False,
        arc_s=64.0,
        arc_m=0.55,
    ):
        super().__init__()
        self.use_arcface = bool(use_arcface)

        weights = models.ResNet34_Weights.IMAGENET1K_V1
        backbone = models.resnet34(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Изначально все слои замораживаю - разморожу layers 3 и 4 на втором этапе обучения
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.3)
        )

        if self.use_arcface:
            self.classifier = ArcFace(in_features=embedding_size, out_features=num_classes, s=arc_s, m=arc_m)
        else:
            self.classifier = nn.Linear(embedding_size, num_classes)
            # Тут используем такую же инициализацию весов, как в arcface
            nn.init.xavier_uniform_(self.classifier.weight)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x, labels=None):
        feats = self.backbone(x)
        embeddings = self.embedding(feats)
        
        if self.use_arcface:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            logits_or_cos = self.classifier(embeddings, labels)
            return logits_or_cos, embeddings
        else:
            logits = self.classifier(embeddings)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return logits, embeddings