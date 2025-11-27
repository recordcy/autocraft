from PIL import Image
import numpy as np
from collections import deque
import random

# ------------------ 팔레트 정의 ------------------ #
PALETTES = {
    "warm_pastel": [
        (255,210,216),(255,233,228),(255,221,210),(255,235,204),
        (255,250,202),(240,255,210),(217,255,217),(204,255,229),
        (214,245,255),(224,224,255),(233,204,255),(255,204,245),
        (255,214,235),(255,194,215),(255,194,194),(255,175,155),
        (255,205,190),(255,220,200),(255,230,210),(250,200,190),
    ],
    "cool_pastel": [
        (204,229,255),(189,224,254),(185,211,238),(195,226,245),
        (198,233,233),(183,230,216),(174,217,200),(190,227,213),
        (200,242,213),(200,255,230),(223,255,228),(220,245,226),
        (200,245,245),(180,240,255),(160,225,250),(140,210,240),
        (210,235,250),(190,230,250),(175,220,245),(160,210,240),
    ],
    "vivid_pastel": [
        (255,99,132),(255,159,64),(255,205,86),(75,192,192),
        (54,162,235),(153,102,255),(255,0,127),(0,200,83),
        (255,111,0),(0,176,255),(255,128,170),(255,180,220),
        (255,140,160),(255,190,120),(255,240,120),(180,240,180),
        (140,210,240),(200,150,250),(250,150,200),(250,200,150),
    ],
    "soft": [
        (200,200,200),(180,180,180),(190,185,175),(210,205,198),
        (189,183,201),(183,200,196),(205,198,189),(190,194,201),
        (200,207,204),(222,220,210),(210,219,216),(170,168,158),
        (195,197,207),(210,212,203),(192,201,197),(199,205,213),
        (185,190,185),(210,210,220),(200,205,210),(190,195,200),
    ],
    "retro": [
        (240,230,180),(225,205,150),(200,180,140),(170,140,110),
        (140,115,90),(200,155,130),(210,180,140),(160,140,120),
        (150,170,140),(140,155,130),(100,135,125),(90,110,105),
        (180,200,190),(210,220,210),(155,180,170),(120,150,135),
        (210,200,170),(190,175,150),(170,150,135),(150,135,120),
    ],
    "neon": [
        (255,0,102),(255,51,0),(255,255,0),(0,255,0),
        (0,255,255),(0,102,255),(153,0,255),(255,0,204),
        (255,102,255),(255,153,102),(255,255,102),(204,255,102),
        (51,255,153),(51,204,255),(102,102,255),(255,102,204),
    ],
    "oriental_ink": [
        (15,15,15),(40,40,40),(65,65,65),(90,90,90),
        (120,120,120),(180,180,180),(210,210,210),(235,235,235),
        (45,30,20),(80,60,45),(110,90,75),(145,130,110),
        (180,150,120),(200,170,140),(220,200,170),(240,230,210),
    ],
    "pokemon": [
        (255,203,5),(255,85,90),(70,128,247),(110,200,140),
        (255,130,50),(130,90,210),(240,240,240),(255,220,0),
        (255,255,155),(104,144,240),(255,184,76),(255,105,97),
        (245,194,255),(148,204,255),(135,175,255),(255,194,184),
    ],
    "choco_cream": [
        (255,244,219),(240,224,194),(225,210,170),(210,195,150),
        (185,160,125),(160,130,100),(135,105,80),(110,80,60),
        (90,60,45),(70,45,35),(55,35,30),(240,230,200),
        (200,180,155),(220,205,180),(180,160,135),(150,125,100),
    ],
    "sunny": [
        (255,255,230),(255,250,210),(255,245,180),(255,240,165),
        (255,230,140),(255,215,120),(255,205,105),(255,195,90),
        (255,185,80),(255,175,70),(255,160,55),(255,145,40),
        (255,230,200),(255,210,180),(255,190,155),(255,165,130),
    ],
    "pastel_rainbow": [
        (255,179,186),(255,223,186),(255,255,186),(186,255,201),
        (186,225,255),(210,190,255),(255,190,225),(255,220,255),
        (200,230,255),(225,255,220),(240,230,255),(255,245,200),
    ],
    "forest_pastel": [
        (204,232,207),(174,200,167),(144,175,140),(124,160,120),
        (104,140,100),(84,120,85),(65,105,70),(50,85,60),
        (170,195,180),(195,215,200),(220,235,220),(240,250,235),
    ],
}

DEFAULT_PALETTE = "warm_pastel"
PALETTES["default"] = PALETTES[DEFAULT_PALETTE]


def dilate_wall(wall: np.ndarray) -> np.ndarray:
    h, w = wall.shape
    seg_wall = wall.copy()
    directions = [
        (0, 0),
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    for dy, dx in directions:
        if dy == 0 and dx == 0:
            continue
        ny1 = max(0, -dy)
        ny2 = min(h, h - dy)
        nx1 = max(0, -dx)
        nx2 = min(w, w - dx)
        seg_wall[ny1:ny2, nx1:nx2] |= wall[ny1+dy:ny2+dy, nx1+dx:nx2+dx]
    return seg_wall


# ------------------ 메인 엔진 ------------------ #
def auto_color_single_image(
    input_path: str,
    output_path: str,
    palette_name: str = "default",
    use_guide: bool = True,
) -> str:
    # 1) 이미지 로드 (원본 크기 그대로)
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # 2) 선(벽) 검출
    r = img_np[..., 0].astype(np.float32)
    g = img_np[..., 1].astype(np.float32)
    b = img_np[..., 2].astype(np.float32)

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    brightness = maxc
    saturation = maxc - minc

    # 검은/짙은 회색 + 채도 낮은 부분을 선으로 간주
    wall = (brightness < 110) & (saturation < 40)
    seg_wall = dilate_wall(wall)   

    # 3) 가이드 마스크 
    dist_from_white = np.sqrt(
        (255 - r) ** 2 +
        (255 - g) ** 2 +
        (255 - b) ** 2
    )
    guide_mask = (~seg_wall) & (dist_from_white > 25.0) & (saturation > 10)

    if use_guide and np.sum(guide_mask) == 0:
        use_guide = False

    # 4) flood-fill 로 영역 분리
    visited = np.zeros((h, w), dtype=bool)
    regions = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or seg_wall[y, x]:
                continue

            q = deque()
            q.append((y, x))
            visited[y, x] = True

            region_pixels = []
            guide_colors = []
            touches_border = False

            while q:
                cy, cx = q.popleft()
                region_pixels.append((cy, cx))

                if cy == 0 or cx == 0 or cy == h - 1 or cx == w - 1:
                    touches_border = True

                if use_guide and guide_mask[cy, cx]:
                    cr, cg, cb = img_np[cy, cx]
                    guide_colors.append((cr, cg, cb))

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < h and 0 <= nx < w and
                        (not visited[ny, nx]) and (not seg_wall[ny, nx])
                    ):
                        visited[ny, nx] = True
                        q.append((ny, nx))

            regions.append(
                {
                    "pixels": region_pixels,
                    "guide_colors": guide_colors,
                    "touch_border": touches_border,
                }
            )

    # 5) 영역 통계 → 배경 파악
    region_has_guide = []
    region_mean_brightness = []
    region_mean_dist = []
    region_touches_border = []

    for region in regions:
        pixels = region["pixels"]
        guide_colors = region["guide_colors"]

        region_has_guide.append(len(guide_colors) > 0)

        br_vals = [brightness[yy, xx] for (yy, xx) in pixels]
        dist_vals = [dist_from_white[yy, xx] for (yy, xx) in pixels]

        region_mean_brightness.append(float(np.mean(br_vals)) if br_vals else 0.0)
        region_mean_dist.append(float(np.mean(dist_vals)) if dist_vals else 999.0)
        region_touches_border.append(region["touch_border"])

    bg_brightness_thresh = 230.0
    bg_dist_max = 50.0
    region_is_background = []

    if use_guide:
        for has_g, mean_b, mean_d, touch in zip(
            region_has_guide,
            region_mean_brightness,
            region_mean_dist,
            region_touches_border,
        ):
            is_bg = (
                (not has_g) and touch and (mean_b > bg_brightness_thresh) and (mean_d < bg_dist_max)
            ) or (
                (not has_g) and (mean_b > 245.0) and (mean_d < 30.0)
            )
            region_is_background.append(is_bg)
    else:
        for mean_b, mean_d, touch in zip(
            region_mean_brightness,
            region_mean_dist,
            region_touches_border,
        ):
            is_bg = touch and (mean_b > bg_brightness_thresh) and (mean_d < bg_dist_max + 20.0)
            region_is_background.append(is_bg)

    # 최소 한 개는 배경으로 지정
    if not any(region_is_background) and len(regions) > 0:
        sizes = [len(r["pixels"]) for r in regions]
        touches = [r["touch_border"] for r in regions]
        candidate_indices = [i for i, t in enumerate(touches) if t]
        if candidate_indices:
            best_idx = max(candidate_indices, key=lambda i: sizes[i])
        else:
            best_idx = int(np.argmax(sizes))
        region_is_background[best_idx] = True

    # 6) 결과 캔버스 초기화 (전체 흰색)
    result = np.full((h, w, 3), 255, dtype=np.uint8)

    palette = list(PALETTES.get(palette_name, PALETTES["default"]))
    random.shuffle(palette)
    palette_index = 0

    # 7) 각 영역 채색 (기본 채우기)
    for idx, region in enumerate(regions):
        pixels = region["pixels"]
        guide_colors = region["guide_colors"]

        if region_is_background[idx]:
            fill_color = (255, 255, 255)  
        else:
            if use_guide and len(guide_colors) > 0:
                avg = np.mean(guide_colors, axis=0).astype(int)
                fill_color = (int(avg[0]), int(avg[1]), int(avg[2]))
            elif use_guide:
                # 가이드 모드인데 가이드 없는 영역 → 비워두기
                fill_color = (255, 255, 255)
            else:
                fill_color = palette[palette_index % len(palette)]
                palette_index += 1

        # 진짜 선(wall) 아닌 픽셀만 색칠
        for (yy, xx) in pixels:
            if not wall[yy, xx]:
                result[yy, xx] = fill_color

    # 8) 선(벽)은 원본 그대로 덮어씀 
    result[wall] = img_np[wall]

    # 9) 선 주변 얇은 흰 틈 메우기 (seg_wall 이지만 wall 은 아닌 픽셀만)
    candidate = seg_wall & (~wall)
    ys, xs = np.where(candidate)

    for y, x in zip(ys, xs):
        # 이미 색이 들어있으면 패스
        if not np.all(result[y, x] > 245):
            continue

        neighbors = []
        for ny in range(max(0, y-1), min(h, y+2)):
            for nx in range(max(0, x-1), min(w, x+2)):
                if ny == y and nx == x:
                    continue
                if wall[ny, nx]:
                    continue
                if not np.all(result[ny, nx] > 245):
                    neighbors.append(result[ny, nx])

        # 주변에 색이 1개 이상 있으면 → 그 평균으로 채움
        if neighbors:
            avg = np.mean(np.array(neighbors), axis=0).astype(np.uint8)
            result[y, x] = avg

    # 10) 저장
    out_img = Image.fromarray(result)
    out_img.save(output_path)
    return output_path


if __name__ == "__main__":
    print("테스트 실행 준비 완료")
    # auto_color_single_image("macaron_guide.png", "macaron_result.png")
