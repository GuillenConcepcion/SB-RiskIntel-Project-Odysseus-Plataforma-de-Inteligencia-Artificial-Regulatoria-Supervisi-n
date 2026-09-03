"""Generate publication-grade LinkedIn Carousel PDF and 2x2 composite infographic for Project Odysseus."""

import logging
import shutil
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from src.config.settings import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

IMAGES_DIR = PROJECT_ROOT / "images"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Color Palette (Institutional SupTech)
NAVY = "#0d1b2a"
DEEP_BLUE = "#1d3557"
TEAL = "#2a9d8f"
CORAL = "#e63946"
GOLD = "#e9c46a"
LIGHT_BG = "#f8f9fa"
DARK_TEXT = "#1a1a1a"


def create_slide_cover(fig):
    """Slide 1: Executive Cover Slide."""
    ax = fig.add_subplot(111)
    ax.set_facecolor(NAVY)
    ax.axis("off")

    # Header Badge
    ax.text(0.5, 0.88, "SUPERINTENDENCIA DE BANCOS - REPUBLICA DOMINICANA", ha="center", va="center", color=GOLD, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.82, "PROYECTO ODYSSEUS | SUPTECH & MLOPS", ha="center", va="center", color="#a8dadc", fontsize=10, fontweight="bold", transform=ax.transAxes)

    # Main Title
    ax.text(0.5, 0.65, "SB-RiskIntel", ha="center", va="center", color="#ffffff", fontsize=34, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.54, "Inteligencia Artificial Regulatoria,\nSupervision Conductual & Alerta Temprana", ha="center", va="center", color="#f1faee", fontsize=17, fontweight="bold", transform=ax.transAxes)

    # Subtitle
    ax.text(0.5, 0.40, "De la Supervision Reactiva a la Analitica Predictiva & Estocastica\n(Series Temporales 2017-2026 - 14 Datasets Oficiales)", ha="center", va="center", color="#e0e1dd", fontsize=11, style="italic", transform=ax.transAxes)

    # Key Highlights Boxes
    highlights = [
        "Conformal Prediction 95%",
        "Monte Carlo N=10,000 (Cholesky)",
        "XAI con SHAP (LMYF 183-02)",
        "Data Drift Monitor (PSI)",
    ]
    for i, hl in enumerate(highlights):
        x = 0.28 + (i % 2) * 0.44
        y = 0.26 - (i // 2) * 0.08
        ax.text(x, y, hl, ha="center", va="center", color="#ffffff", fontsize=10, fontweight="bold", bbox=dict(boxstyle="round,pad=0.5", facecolor=DEEP_BLUE, edgecolor=TEAL, alpha=0.9), transform=ax.transAxes)

    # Author Footer
    ax.text(0.5, 0.08, "Por: Guillen Concepcion | Senior Data Scientist & MLOps Engineer", ha="center", va="center", color="#a8dadc", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.04, "github.com/GuillenConcepcion/SB-RiskIntel-Project-Odysseus", ha="center", va="center", color=GOLD, fontsize=9, transform=ax.transAxes)


def create_slide_with_image(fig, img_path: Path, slide_num: int, title: str, subtitle: str, key_takeaways: list):
    """Generic Slide: Title + High-Res Image + Key Quantitative Takeaways."""
    # Top banner & Title
    ax_top = fig.add_axes([0.05, 0.86, 0.90, 0.12])
    ax_top.set_facecolor(NAVY)
    ax_top.axis("off")

    ax_top.text(0.02, 0.70, f"SLIDE {slide_num:02d} / 05 | {title}", color="#ffffff", fontsize=14, fontweight="bold", transform=ax_top.transAxes)
    ax_top.text(0.02, 0.25, subtitle, color="#a8dadc", fontsize=9.5, transform=ax_top.transAxes)
    ax_top.text(0.98, 0.50, "SB-RiskIntel", ha="right", color=GOLD, fontsize=11, fontweight="bold", transform=ax_top.transAxes)

    # Image Area
    if img_path.exists():
        ax_img = fig.add_axes([0.05, 0.26, 0.90, 0.58])
        img = Image.open(img_path)
        ax_img.imshow(img)
        ax_img.axis("off")

    # Bottom Takeaways Footer
    ax_bot = fig.add_axes([0.05, 0.04, 0.90, 0.20])
    ax_bot.set_facecolor(LIGHT_BG)
    ax_bot.axis("off")

    ax_bot.text(0.02, 0.85, "Hallazgos Cuantitativos & Relevancia Regulatoria:", color=DEEP_BLUE, fontsize=10, fontweight="bold", transform=ax_bot.transAxes)
    for idx, point in enumerate(key_takeaways):
        ax_bot.text(0.02, 0.55 - idx * 0.32, f"- {point}", color=DARK_TEXT, fontsize=9.5, transform=ax_bot.transAxes)


def create_slide_closing(fig):
    """Slide 6: Conclusion and Call to Action (CTA)."""
    ax = fig.add_subplot(111)
    ax.set_facecolor(NAVY)
    ax.axis("off")

    ax.text(0.5, 0.88, "SUPERINTENDENCIA DE BANCOS DE LA REPUBLICA DOMINICANA", ha="center", va="center", color=GOLD, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.76, "Transformando la Supervision Financiera con IA", ha="center", va="center", color="#ffffff", fontsize=22, fontweight="bold", transform=ax.transAxes)

    benefits = [
        ("Proteccion Proactiva:", "Dimensionamiento de reservas con certidumbre estadistica finita (Conformal 95%)."),
        ("Resiliencia Sistemica:", "Simulacion Monte Carlo (N=10,000) bajo choques correlacionados multivariados."),
        ("Debida Motivacion Legal:", "Explicabilidad XAI (SHAP & PDP) para cumplimiento estricto de la Ley 183-02."),
        ("Excelencia en Ingenieria:", "Arquitectura Cloud-Native, 31 tests automatizados, cache LRU/Redis y MLOps."),
    ]

    for i, (b_title, b_desc) in enumerate(benefits):
        y = 0.60 - i * 0.10
        ax.text(0.15, y, b_title, color=GOLD, fontsize=12, fontweight="bold", transform=ax.transAxes)
        ax.text(0.15, y - 0.035, b_desc, color="#f1faee", fontsize=10, transform=ax.transAxes)

    # CTA Box
    ax.text(0.5, 0.16, "Codigo Abierto, Modelos & Pipeline Completo en GitHub:", ha="center", va="center", color="#a8dadc", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.10, "https://github.com/GuillenConcepcion/SB-RiskIntel-Project-Odysseus", ha="center", va="center", color="#ffffff", fontsize=10.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.6", facecolor=CORAL, edgecolor="none"), transform=ax.transAxes)
    ax.text(0.5, 0.04, "Autor: Guillen Concepcion | Senior Data Scientist & MLOps Engineer", ha="center", va="center", color="#e0e1dd", fontsize=9, transform=ax.transAxes)


def generate_linkedin_pdf_carousel():
    """Build multi-page LinkedIn Carousel PDF (16:10 aspect ratio, 300 DPI)."""
    pdf_path_images = IMAGES_DIR / "SB_RiskIntel_LinkedIn_Carousel.pdf"
    pdf_path_docs = DOCS_DIR / "SB_RiskIntel_LinkedIn_Carousel.pdf"

    slides_config = [
        {
            "img": IMAGES_DIR / "01_conformal_prediction_intervals.png",
            "title": "Inferencia Conforme (Conformal Prediction 90% & 95%)",
            "subtitle": "Bandas de Cobertura Garantizada para Reclamaciones y Restitucion Monetaria (DOP)",
            "takeaways": [
                "Garantia estadistica finita P(Y en [y_low, y_upp]) >= 1 - alpha sin asumir normalidad (Distribution-Free).",
                "Permite al regulador y a las EIF dimensionar buffers de restitucion para ProUsuario con certidumbre no parametrica.",
            ],
        },
        {
            "img": IMAGES_DIR / "02_monte_carlo_stress_distribution.png",
            "title": "Simulador de Estres Estocastico Monte Carlo (N=10,000)",
            "subtitle": "Distribucion Multivariada con Factor de Cholesky bajo Choques Macroeconomicos",
            "takeaways": [
                "Descomposicion Cholesky que preserva la correlacion empirica entre variables de friccion e infracciones.",
                "Cuantificacion exacta de colas criticas: VaR 95% = DOP 327.2M | CVaR 95% (Expected Shortfall) = DOP 359.2M.",
            ],
        },
        {
            "img": IMAGES_DIR / "03_xai_shap_feature_importance.png",
            "title": "Explicabilidad XAI Regulatoria (SHAP Global Feature Importance)",
            "subtitle": "Debida Motivacion Tecnica y Legal de Alertas Tempranas (Ley Monetaria y Financiera 183-02)",
            "takeaways": [
                "Atribucion cuantitativa por TreeExplainer (mean |SHAP|): sanciones previas y solicitudes AML dominan el riesgo.",
                "Fundamentacion transparente de actos administrativos e inspecciones in situ ante el marco legal bancario.",
            ],
        },
        {
            "img": IMAGES_DIR / "05_supervisory_latent_clusters_pca.png",
            "title": "Espacio Latente PCA & Arquetipos de Riesgo Conductual",
            "subtitle": "Segmentacion No Supervisada de Entidades de Intermediacion Financiera (Silhouette Opt)",
            "takeaways": [
                "Optimizacion por Silhouette Score (k=2) proyectada en las dos componentes principales de supervision.",
                "Diferenciacion clara entre 'Supervision Estable' vs. 'Alta Intensidad Sancionadora / Friccion Activa'.",
            ],
        },
    ]

    logger.info(f"Generating LinkedIn Carousel PDF at {pdf_path_images}...")

    with PdfPages(pdf_path_images) as pdf:
        # Slide 1: Cover
        fig_cover = plt.figure(figsize=(10, 8), dpi=300)
        create_slide_cover(fig_cover)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        # Slides 2-5: Core Visualizations
        for idx, s in enumerate(slides_config, start=2):
            fig_slide = plt.figure(figsize=(10, 8), dpi=300)
            create_slide_with_image(fig_slide, s["img"], idx, s["title"], s["subtitle"], s["takeaways"])
            pdf.savefig(fig_slide)
            plt.close(fig_slide)

        # Slide 6: Closing
        fig_close = plt.figure(figsize=(10, 8), dpi=300)
        create_slide_closing(fig_close)
        pdf.savefig(fig_close)
        plt.close(fig_close)

    # Copy to docs/
    shutil.copy2(pdf_path_images, pdf_path_docs)
    logger.info(f"LinkedIn Carousel PDF successfully generated ({pdf_path_images.stat().st_size / 1024:.1f} KB)")


def generate_composite_showcase_image():
    """Generate a single 2x2 grid image (300 DPI) for instant preview/sharing."""
    out_img_path = IMAGES_DIR / "00_linkedin_carousel_showcase_grid.png"
    logger.info("Generating 2x2 Composite Showcase Image...")

    img1 = Image.open(IMAGES_DIR / "01_conformal_prediction_intervals.png")
    img2 = Image.open(IMAGES_DIR / "02_monte_carlo_stress_distribution.png")
    img3 = Image.open(IMAGES_DIR / "03_xai_shap_feature_importance.png")
    img5 = Image.open(IMAGES_DIR / "05_supervisory_latent_clusters_pca.png")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor("#f8f9fa")

    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("1. Inferencia Conforme (90% & 95% Cobertura)", fontsize=13, fontweight="bold", color=DEEP_BLUE, pad=8)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("2. Simulacion de Estres Monte Carlo (N=10,000 Cholesky)", fontsize=13, fontweight="bold", color=DEEP_BLUE, pad=8)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(img3)
    axes[1, 0].set_title("3. Explicabilidad XAI SHAP (LMYF 183-02)", fontsize=13, fontweight="bold", color=DEEP_BLUE, pad=8)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img5)
    axes[1, 1].set_title("4. Espacio Latente PCA & Arquetipos Conductuales", fontsize=13, fontweight="bold", color=DEEP_BLUE, pad=8)
    axes[1, 1].axis("off")

    plt.suptitle("SB-RiskIntel (Project Odysseus) | Portafolio Visual SupTech & Inteligencia Artificial Regulatoria", fontsize=17, fontweight="bold", color=NAVY, y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(out_img_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info(f"Composite showcase image generated at {out_img_path}")


if __name__ == "__main__":
    generate_linkedin_pdf_carousel()
    generate_composite_showcase_image()
    print(">>> All LinkedIn Carousel assets (PDF + 2x2 Showcase Grid) generated successfully!", flush=True)
