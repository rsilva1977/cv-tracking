# Object Motion Tracker

Script em Python com OpenCV para rastreio de objetos em movimento em tempo real, através de webcam ou ficheiros de vídeo.

## Funcionalidades

- **Modo automático (`motion`)**: Deteta e acompanha o principal objeto em movimento através de subtração de fundo (`MOG2`) combinada com um **Filtro de Kalman 2D** para suavizar o movimento e prever a posição em pequenas oclusões.
- **Modo seleção manual (`select`)**: Permite desenhar um retângulo à volta de um objeto específico e rastreá-lo continuamente (`TrackerMIL`).
- **HUD em tempo real**: Caixa delimitadora, centroide, rasto da trajetória, vetor de direção/velocidade, coordenadas `(x, y)` e FPS.
- **Suporte flexível**: Funciona com webcam, ficheiros de vídeo e modo *headless* (apenas terminal).
- **Compatibilidade Wayland (Linux)**: Configura automaticamente `QT_QPA_PLATFORM=xcb` quando executado em sessões Wayland.

## Requisitos

- Python 3.10 ou superior
- Webcam (opcional — também aceita ficheiros de vídeo)

## Instalação

```bash
# Clonar o repositório
git clone <URL_DO_REPOSITORIO>
cd <PASTA_DO_REPOSITORIO>

# Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Como usar

```bash
# Rastreio automático com a webcam (modo por defeito)
python tracker.py

# Selecionar manualmente o objeto a rastrear com o rato
python tracker.py --mode select

# Rastrear num ficheiro de vídeo
python tracker.py --video caminho/para/video.mp4

# Ajustar câmara, sensibilidade (área mínima) e tamanho do rasto
python tracker.py --camera 0 --min-area 800 --trail 60

# Executar sem janela gráfica (apenas coordenadas no terminal)
python tracker.py --headless
```

### Opções de linha de comandos

| Opção | Descrição | Valor por defeito |
| --- | --- | --- |
| `--video` | Caminho para um ficheiro de vídeo (se omitido, usa a webcam) | `None` |
| `--camera` | Índice da webcam | `0` |
| `--mode` | Modo inicial: `motion` (automático) ou `select` (manual) | `motion` |
| `--min-area` | Área mínima em píxeis para detetar movimento no modo automático | `1200` |
| `--trail` | Número de pontos mantidos no rasto da trajetória | `40` |
| `--no-flip` | Desativa o espelhamento horizontal da webcam | `False` |
| `--headless` | Executa sem interface gráfica | `False` |

### Atalhos de teclado (na janela de vídeo)

| Tecla | Ação |
| --- | --- |
| `s` | Selecionar um objeto manualmente com o rato |
| `m` | Voltar ao modo automático por deteção de movimento |
| `r` | Reiniciar deteção de fundo e limpar o rasto |
| `q` ou `ESC` | Sair |

## Estrutura do projeto

```text
.
├── tracker.py        # Script principal de rastreio
├── requirements.txt  # Dependências Python
├── README.md         # Documentação
├── LICENSE           # Licença MIT
└── .gitignore        # Ficheiros ignorados pelo Git
```

## Licença

Distribuído sob a licença MIT. Consulta o ficheiro [LICENSE](LICENSE) para mais detalhes.
