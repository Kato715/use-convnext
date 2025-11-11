import torch.nn as nn
import torchvision as tv


def select(model_type: str, num_classes: int) -> nn.Module:
    model_list = {
        "resnet18": Resnet18,
        "resnet152": Resnet152,
        "vgg11": VGG11,
        "efficientnetb0": EfficientNetB0,
        "efficientnetb7": EfficientNetB7,
        "convnext-tiny": ConvnextTiny,
        "convnext-small": ConvnextSmall,
        "convnext-base": ConvnextBase,
        "convnext-large": ConvnextLarge,
    }

    if model_type not in model_list:
        raise ValueError(f"Unsupported model type.")

    return model_list[model_type](num_classes)


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()

        self.convnet = tv.models.resnet18(
            weights=tv.models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.convnet.fc = nn.Linear(512, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.convnet(image)

        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss

        return outputs, None


class Resnet152(nn.Module):
    def __init__(self, num_classes):
        super(Resnet152, self).__init__()

        self.convnet = tv.models.resnet152(
            weights=tv.models.ResNet152_Weights.IMAGENET1K_V1
        )

        self.convnet.fc = nn.Linear(2048, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.convnet(image)

        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss

        return outputs, None


class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()

        self.base = tv.models.vgg11(weights=tv.models.VGG11_Weights.IMAGENET1K_V1)

        self.base.classifier[6] = nn.Linear(4096, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)

        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss

        return outputs, None


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()

        self.base = tv.models.efficientnet_b0(
            weights=tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        self.base.classifier[1] = nn.Linear(1280, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)

        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss

        return outputs, None


class EfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7, self).__init__()

        self.base = tv.models.efficientnet_b7(
            weights=tv.models.EfficientNet_B7_Weights.IMAGENET1K_V1
        )

        self.base.classifier[1] = nn.Linear(2560, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)

        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss

        return outputs, None


class ConvnextTiny(nn.Module):
    def __init__(self, num_classes):
        super(ConvnextTiny, self).__init__()

        self.base = tv.models.convnext_tiny(
            weights=tv.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        self.base.classifier[2] = nn.Linear(768, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)

        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss

        return outputs, None


class ConvnextSmall(nn.Module):
    def __init__(self, num_classes):
        super(ConvnextSmall, self).__init__()

        self.base = tv.models.convnext_small(
            weights=tv.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )

        self.base.classifier[2] = nn.Linear(768, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)

        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss

        return outputs, None


class ConvnextBase(nn.Module):
    def __init__(self, num_classes):
        super(ConvnextBase, self).__init__()

        self.base = tv.models.convnext_base(
            weights=tv.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )

        self.base.classifier[2] = nn.Linear(1024, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class ConvnextLarge(nn.Module):
    def __init__(self, num_classes):
        super(ConvnextLarge, self).__init__()

        self.base = tv.models.convnext_large(
            weights=tv.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        )

        self.base.classifier[2] = nn.Linear(1536, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)

        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)

        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss

        return outputs, None

