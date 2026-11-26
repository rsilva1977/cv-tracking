import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

# OpenCV pip inclui apenas o plugin Qt 'xcb'; evita aviso/erro em sessões Wayland
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np


class KalmanPointTracker:
    """Filtro de Kalman 2D (posição + velocidade) para suavizar o rastreio."""

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.initialized = False

    def reset(self) -> None:
        self.initialized = False

    def predict(self) -> tuple[float, float] | None:
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def update(self, x: float, y: float) -> tuple[int, int, float, float]:
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array(
                [[np.float32(x)], [np.float32(y)], [0.0], [0.0]], dtype=np.float32
            )
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
        else:
            self.kf.predict()
        estimated = self.kf.correct(measurement)
        cx, cy = int(round(float(estimated[0, 0]))), int(round(float(estimated[1, 0])))
        vx, vy = float(estimated[2, 0]), float(estimated[3, 0])
        return cx, cy, vx, vy


class MotionObjectDetector:
    """Deteta e acompanha o principal objeto em movimento via subtração de fundo."""

    def __init__(self, min_area: int = 1200) -> None:
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kalman = KalmanPointTracker()
        self.missed_frames = 0
        self.max_missed = 15
        self.bg = self._create_bg()

    @staticmethod
    def _create_bg():
        return cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=True
        )

    def reset(self) -> None:
        self.bg = self._create_bg()
        self.kalman.reset()
        self.missed_frames = 0

    def detect(
        self, frame: np.ndarray
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None, tuple[float, float]]:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = self.bg.apply(blurred)

        # Remover sombras (valor 127 no MOG2) e ruído pequeno
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w / 2.0
            cy = y + h / 2.0
            candidates.append((area, (x, y, w, h), (cx, cy)))

        if not candidates:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed:
                self.kalman.reset()
                return None, None, (0.0, 0.0)
            pred = self.kalman.predict()
            if pred is None:
                return None, None, (0.0, 0.0)
            return None, (int(round(pred[0])), int(round(pred[1]))), (0.0, 0.0)

        self.missed_frames = 0

        # Se já estamos a seguir um objeto, preferir o candidato mais próximo da previsão
        pred = self.kalman.predict() if self.kalman.initialized else None
        if pred is not None:
            px, py = pred
            best = min(
                candidates,
                key=lambda item: ((item[2][0] - px) ** 2 + (item[2][1] - py) ** 2)
                / (item[0] ** 0.25),
            )
        else:
            best = max(candidates, key=lambda item: item[0])

        _, bbox, (raw_cx, raw_cy) = best
        smooth_cx, smooth_cy, vx, vy = self.kalman.update(raw_cx, raw_cy)
        return bbox, (smooth_cx, smooth_cy), (vx, vy)


def draw_hud(
    frame: np.ndarray,
    mode: str,
    bbox: tuple[int, int, int, int] | None,
    center: tuple[int, int] | None,
    velocity: tuple[float, float],
    trail: deque[tuple[int, int]],
    fps: float,
    status_text: str,
) -> np.ndarray:
    # Desenhar trajetória (rasto)
    points = list(trail)
    n = len(points)
    for i in range(1, n):
        thickness = max(1, int(np.sqrt(float(i + 1) / n) * 4))
        intensity = int(80 + 175 * (i / n))
        cv2.line(frame, points[i - 1], points[i], (0, intensity, 255), thickness, cv2.LINE_AA)

    # Desenhar caixa delimitadora e centroide
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 120), 2)

    if center is not None:
        cx, cy = center
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        # Vetor de direção (ampliado para ser visível)
        vx, vy = velocity
        if abs(vx) > 0.5 or abs(vy) > 0.5:
            end_pt = (int(cx + vx * 4), int(cy + vy * 4))
            cv2.arrowedLine(frame, (cx, cy), end_pt, (255, 200, 0), 2, tipLength=0.3)

    # Painel superior esquerdo
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (370, 96), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    mode_label = "AUTO (Movimento)" if mode == "motion" else "MANUAL (ROI)"
    cv2.putText(
        frame,
        f"Modo: {mode_label}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (285, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    pos_text = f"Pos: ({center[0]}, {center[1]})" if center else "Pos: --"
    cv2.putText(
        frame,
        f"{status_text} | {pos_text}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Teclas: [s] Selecionar  [m] Auto  [r] Reset  [q] Sair",
        (20, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return frame


def run_tracker(
    source: int | str = 0,
    mode: str = "motion",
    flip: bool = True,
    min_area: int = 1200,
    trail_len: int = 40,
    headless: bool = False,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(
            f"Erro: não foi possível abrir a fonte de vídeo ({source}).",
            file=sys.stderr,
        )
        sys.exit(1)

    window_name = "Rastreio de Objeto em Movimento"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = MotionObjectDetector(min_area=min_area)
    roi_tracker = None
    roi_kalman = KalmanPointTracker()
    trail: deque[tuple[int, int]] = deque(maxlen=trail_len)

    def select_object_on_frame(frame: np.ndarray) -> bool:
        nonlocal roi_tracker, mode
        if headless:
            return False
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = [int(v) for v in roi]
        if w > 0 and h > 0:
            roi_tracker = cv2.TrackerMIL_create()
            roi_tracker.init(frame, (x, y, w, h))
            roi_kalman.reset()
            trail.clear()
            mode = "select"
            return True
        return False

    print(
        "Rastreio iniciado. Teclas na janela: "
        "'s' selecionar objeto, 'm' modo automático, 'r' reiniciar, 'q'/ESC sair."
    )

    fps = 0.0
    prev_time = time.perf_counter()
    first_frame_pending_select = mode == "select"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if flip and isinstance(source, int):
                frame = cv2.flip(frame, 1)

            if first_frame_pending_select:
                first_frame_pending_select = False
                if not select_object_on_frame(frame):
                    mode = "motion"

            bbox = None
            center = None
            velocity = (0.0, 0.0)
            status_text = "A aguardar movimento"

            if mode == "motion":
                bbox, center, velocity = detector.detect(frame)
                if bbox is not None:
                    status_text = "Objeto detetado"
                elif center is not None:
                    status_text = "A prever posicao"
            else:
                if roi_tracker is not None:
                    found, raw_bbox = roi_tracker.update(frame)
                    if found:
                        x, y, w, h = [int(v) for v in raw_bbox]
                        bbox = (x, y, w, h)
                        cx, cy, vx, vy = roi_kalman.update(x + w / 2.0, y + h / 2.0)
                        center = (cx, cy)
                        velocity = (vx, vy)
                        status_text = "A rastrear"
                    else:
                        status_text = "Perdido (prime 's')"
                else:
                    status_text = "Prime 's' para selecionar"

            if center is not None:
                trail.append(center)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0.0 else (0.85 * fps + 0.15 * inst_fps)

            pos_str = f"({center[0]:4d}, {center[1]:4d})" if center else "(  --,   --)"
            print(
                f"\r[{mode:6s}] {status_text:20s} | Pos: {pos_str} | {fps:4.1f} FPS   ",
                end="",
                flush=True,
            )

            if not headless:
                display = draw_hud(
                    frame, mode, bbox, center, velocity, trail, fps, status_text
                )
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("s"), ord("S")):
                    select_object_on_frame(frame)
                elif key in (ord("m"), ord("M")):
                    mode = "motion"
                    roi_tracker = None
                    detector.reset()
                    trail.clear()
                elif key in (ord("r"), ord("R")):
                    detector.reset()
                    roi_kalman.reset()
                    trail.clear()

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rastreia um objeto em movimento em tempo real via webcam ou ficheiro de vídeo."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Caminho opcional para um ficheiro de vídeo (se omitido, usa a webcam).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice da webcam (por defeito: 0).",
    )
    parser.add_argument(
        "--mode",
        choices=("motion", "select"),
        default="motion",
        help=(
            "Modo inicial: 'motion' deteta movimento automaticamente; "
            "'select' permite desenhar um retângulo à volta do objeto."
        ),
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1200,
        help="Área mínima em píxeis para considerar movimento no modo automático (defeito: 1200).",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=40,
        help="Número de pontos no rasto da trajetória (defeito: 40).",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Desativa o espelhamento horizontal da webcam.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Executa sem janela gráfica, mostrando as coordenadas no terminal.",
    )
    args = parser.parse_args()

    if args.video is not None:
        if not args.video.is_file():
            print(f"Erro: vídeo não encontrado: {args.video}", file=sys.stderr)
            sys.exit(1)
        source: int | str = str(args.video)
    else:
        source = args.camera

    run_tracker(
        source=source,
        mode=args.mode,
        flip=not args.no_flip,
        min_area=args.min_area,
        trail_len=args.trail,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
