import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

# OpenCV usa Qt para mostrar janelas. Em sessões Wayland do Linux, forçamos o
# plugin "xcb" compatível para evitar erros ao abrir a janela de vídeo.
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np


class KalmanPointTracker:
    """Estima e suaviza a posição 2D usando um filtro de Kalman.

    O estado guardado pelo filtro contém a posição (x, y) e a velocidade
    (vx, vy). Assim, além de reduzir pequenos tremores, o filtro pode estimar
    onde o objeto provavelmente estará durante alguns quadros em que não é
    detetado.
    """

    def __init__(self) -> None:
        # cv2.KalmanFilter(numero_de_variaveis, numero_de_medicoes).
        # 4 variáveis: x, y, vx, vy. 2 medições: x e y.
        self.kf = cv2.KalmanFilter(4, 2)

        # Diz ao filtro que as medições fornecidas correspondem apenas às
        # posições x e y, e não às velocidades.
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )

        # Modelo de movimento simples: a posição seguinte é a posição atual
        # mais a velocidade. Exemplo: novo_x = x + vx.
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Quanto mais baixos os ruídos, mais o filtro confia no modelo. O
        # ruído de medição representa pequenas imprecisões da deteção visual.
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        # Indica se o filtro já recebeu pelo menos uma posição real.
        self.initialized = False

    def reset(self) -> None:
        """Esquece a posição e a velocidade anteriores."""
        self.initialized = False

    def predict(self) -> tuple[float, float] | None:
        """Tenta prever a posição sem receber uma nova deteção."""
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def update(self, x: float, y: float) -> tuple[int, int, float, float]:
        """Combina uma deteção real com a previsão anterior do filtro."""
        # Em OpenCV, uma medição é uma matriz coluna com os valores x e y.
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            # Na primeira deteção não existe velocidade conhecida, por isso
            # começamos com velocidade zero.
            self.kf.statePre = np.array(
                [[np.float32(x)], [np.float32(y)], [0.0], [0.0]], dtype=np.float32
            )
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
        else:
            # Antes de corrigir a previsão, avançamos o filtro um quadro.
            self.kf.predict()

        # "correct" usa a posição real para corrigir a estimativa do filtro.
        estimated = self.kf.correct(measurement)
        cx, cy = int(round(float(estimated[0, 0]))), int(round(float(estimated[1, 0])))
        vx, vy = float(estimated[2, 0]), float(estimated[3, 0])
        return cx, cy, vx, vy


class MotionObjectDetector:
    """Deteta e acompanha o principal objeto em movimento.

    Primeiro aprende o aspeto estático da imagem. Depois identifica como
    movimento qualquer região que deixe de ser semelhante ao fundo. Se vários
    objetos forem encontrados, escolhe o candidato mais próximo da posição
    prevista pelo filtro de Kalman.
    """

    def __init__(self, min_area: int = 1200) -> None:
        # O valor min_area evita que pequenas alterações sejam consideradas objetos.
        self.min_area = min_area
        # Kernel usado pelas operações morfológicas para eliminar ou preencher
        # pequenos grupos de píxeis.
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kalman = KalmanPointTracker()
        # Missed frames conta quantos quadros seguidos não encontraram movimento.
        self.missed_frames = 0
        self.max_missed = 15
        self.bg = self._create_bg()

    @staticmethod
    def _create_bg():
        """Cria o modelo que aprende quais são as partes estáticas do vídeo."""
        return cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=True
        )

    def reset(self) -> None:
        """Recomeça a aprendizagem do fundo e apaga o rastreio anterior."""
        self.bg = self._create_bg()
        self.kalman.reset()
        self.missed_frames = 0

    def detect(
        self, frame: np.ndarray
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None, tuple[float, float]]:
        # A imagem borrada perde detalhes pequenos e fica menos sensível a
        # variações aleatórias de um píxel para o outro.
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # A máscara tem apenas valores diferentes de zero nas áreas consideradas
        # movimento. 255 representa movimento; 0 representa fundo.
        fg_mask = self.bg.apply(blurred)

        # MOG2 usa o valor 127 para sombras. O limiar remove sombras e ruído:
        # somente píxeis claramente diferentes do fundo permanecem em branco.
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # OPEN remove pequenos ruídos isolados.
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        # CLOSE preenche pequenos buracos dentro da região movida.
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        # Dilatação amplia ligeiramente o contorno para evitar que partes dele
        # fiquem separadas em vários pedaços.
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # Cada contorno encontrado é uma área branca ligada na máscara.
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            # area ajuda a rejeitar objetos demasiado pequenos.
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # boundingRect devolve a caixa: posição inicial (x, y), largura e altura.
            x, y, w, h = cv2.boundingRect(cnt)
            # O centro da caixa é usado como posição medida do objeto.
            cx = x + w / 2.0
            cy = y + h / 2.0
            candidates.append((area, (x, y, w, h), (cx, cy)))

        if not candidates:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed:
                # Depois de muitos quadros sem deteção, o filtro pode estar a
                # prever uma posição antiga e é mais seguro reiniciá-lo.
                self.kalman.reset()
                return None, None, (0.0, 0.0)

            # Enquanto o objeto está momentaneamente escondido, devolvemos a
            # posição prevista para manter a interface informada.
            pred = self.kalman.predict()
            if pred is None:
                return None, None, (0.0, 0.0)
            return None, (int(round(pred[0])), int(round(pred[1]))), (0.0, 0.0)

        self.missed_frames = 0

        # Com um rastreio ativo, preferimos o candidato mais próximo da previsão.
        # Isso ajuda a não trocar de objeto apenas porque apareceu outro em
        # movimento. Quando não existe previsão, escolhemos a maior área.
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
    """Desenha a informação visual por cima do quadro do vídeo.

    HUD significa "head-up display": são textos e formas desenhadas diretamente
    sobre a imagem, sem alterar os dados usados pelo rastreio.
    """
    # Percorre os pontos guardados no rasto em sequência e liga cada par com
    # uma linha. Os pontos mais recentes são mais grossos e mais intensos.
    points = list(trail)
    n = len(points)
    for i in range(1, n):
        thickness = max(1, int(np.sqrt(float(i + 1) / n) * 4))
        intensity = int(80 + 175 * (i / n))
        cv2.line(frame, points[i - 1], points[i], (0, intensity, 255), thickness, cv2.LINE_AA)

    # Uma caixa verde mostra a região ocupada pelo objeto.
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 120), 2)

    # Um círculo amarelo marca o centro do objeto. A seta amarela representa
    # a direção do movimento calculada pelo filtro.
    if center is not None:
        cx, cy = center
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        vx, vy = velocity
        if abs(vx) > 0.5 or abs(vy) > 0.5:
            end_pt = (int(cx + vx * 4), int(cy + vy * 4))
            cv2.arrowedLine(frame, (cx, cy), end_pt, (255, 200, 0), 2, tipLength=0.3)

    # Primeiro criamos uma versão escura da área do painel e misturamos essa
    # camada com o quadro original para garantir legibilidade do texto.
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
    """Abre a origem de vídeo e executa o ciclo principal do programa.

    Cada repetição deste método representa um quadro:
    1. ler a imagem;
    2. detetar ou seguir o objeto;
    3. atualizar o rasto e o FPS;
    4. mostrar ou imprimir o resultado;
    5. processar teclas do utilizador.
    """
    # VideoCapture consegue abrir tanto uma webcam (indice inteiro) quanto
    # um ficheiro de vídeo (caminho em texto).
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

    # Detector MOG2 é usado no modo automático.
    detector = MotionObjectDetector(min_area=min_area)
    # O rastreador MIL só existe depois de o utilizador selecionar um retângulo.
    roi_tracker = None
    # Este filtro suaviza as posições obtidas pelo rastreador manual.
    roi_kalman = KalmanPointTracker()
    # deque elimina automaticamente os pontos mais antigos quando a capacidade
    # máxima (trail_len) é atingida.
    trail: deque[tuple[int, int]] = deque(maxlen=trail_len)

    def select_object_on_frame(frame: np.ndarray) -> bool:
        nonlocal roi_tracker, mode
        if headless:
            return False

        # selectROI pausa a execução até o utilizador desenhar o retângulo.
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = [int(v) for v in roi]
        if w > 0 and h > 0:
            # O rastreador MIL recebe a primeira imagem e a região selecionada.
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
            # cap.read devolve ok=False quando o vídeo termina ou a leitura falha.
            ok, frame = cap.read()
            if not ok:
                break

            # Espelhamos a webcam para parecer uma "vista de espelho".
            # Em ficheiros de vídeo, o espelhamento não é aplicado.
            if flip and isinstance(source, int):
                frame = cv2.flip(frame, 1)

            # Esta condição permite selecionar imediatamente o objeto quando
            # --mode select é usado na inicialização.
            if first_frame_pending_select:
                first_frame_pending_select = False
                if not select_object_on_frame(frame):
                    mode = "motion"

            # Valores iniciais neutros; são substituídos quando algo é encontrado.
            bbox = None
            center = None
            velocity = (0.0, 0.0)
            status_text = "A aguardar movimento"

            if mode == "motion":
                # Detecta movimento em cada novo quadro.
                bbox, center, velocity = detector.detect(frame)
                if bbox is not None:
                    status_text = "Objeto detetado"
                elif center is not None:
                    status_text = "A prever posicao"
            else:
                if roi_tracker is not None:
                    # update tenta localizar o objeto selecionado no novo quadro.
                    found, raw_bbox = roi_tracker.update(frame)
                    if found:
                        x, y, w, h = [int(v) for v in raw_bbox]
                        bbox = (x, y, w, h)
                        # A posição usada pelo filtro é o centro da caixa obtida.
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

            # Medir o tempo entre dois quadros permite estimar quantos quadros
            # são processados por segundo (FPS).
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                # A média suavizada evita que oscilações momentâneas alterem
                # demasiado o valor mostrado.
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

                # waitKey procura uma tecla durante 1 milissegundo.
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
    """Lê os argumentos da linha de comandos e inicia o rastreador."""
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
