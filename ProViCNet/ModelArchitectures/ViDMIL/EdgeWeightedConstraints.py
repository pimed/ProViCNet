import torch
import torch.nn.functional as F

def generate_edge_mask(LabelMap, edge_threshold=0.5):
    """
    LabelMap: 세그멘테이션 결과 맵 (Tensor, shape: [N, H, W])
    edge_threshold: 엣지 감지 임계값
    """
    # Sobel 연산자 정의
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0)
    
    if LabelMap.is_cuda:
        sobel_x, sobel_y = sobel_x.cuda(), sobel_y.cuda()

    # Sobel 필터 적용을 위한 입력 타입을 FloatTensor로 변환
    LabelMap_float = LabelMap.unsqueeze(1).float()  # 입력을 float 타입으로 변환

    # Sobel 필터 적용
    edge_x = F.conv2d(LabelMap_float, sobel_x, padding=1)
    edge_y = F.conv2d(LabelMap_float, sobel_y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2)

    # 임계값을 사용하여 엣지| 마스크 생성
    edge_mask = (edge > edge_threshold).float()

    return edge_mask


def dilate_edge_mask(edge_mask, dilation_rate=3):
    """
    edge_mask: 엣지 감지 결과 마스크 (Tensor, shape: [N, 1, H, W])
    dilation_rate: 엣지를 확장할 비율
    """
    # 엣지 마스크에 대한 dilation 적용
    dilated_edge_mask = F.max_pool2d(edge_mask, kernel_size=dilation_rate, stride=1, padding=dilation_rate//2)
    return dilated_edge_mask

def total_variation_loss(img, variable_weights):
    """
    img: 세그멘테이션 결과 이미지 (Tensor, shape: [N, C, H, W])
    variable_weights: 가변적 가중치 (Tensor, shape: [N, 1, H, W])
    """
    # TV Loss 계산
    pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]
    pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]
    
    # 가변적 가중치 적용
    tv_loss = (torch.sum(torch.abs(pixel_dif1) * variable_weights[:, :, :-1, :]) +
               torch.sum(torch.abs(pixel_dif2) * variable_weights[:, :, :, :-1]))
    
    return tv_loss
