"""
Literature Review Paper Downloader
====================================
Downloads all papers cited in the MBE generation-stage experiment literature review.
Papers are saved to ./papers/ with clean filenames.

Sources used (in priority order):
  1. arXiv PDF (arxiv.org/pdf/{id})
  2. Direct PDF URL (where known)
  3. Semantic Scholar / unpaywall (fallback)

Usage:
  pip install requests tqdm
  python download_papers.py
  python download_papers.py --output my_folder
  python download_papers.py --dry-run        # print URLs only, no download
"""

import os
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# PAPER REGISTRY
# Each entry: (filename_stem, url, citation_key, full_reference)
# url: prefer arXiv PDF link; fall back to direct PDF where available
# ──────────────────────────────────────────────────────────────────────────────

PAPERS = [

    # ── 1. Core Legal Hallucination Papers ────────────────────────────────────

    (
        "dahl_2024_large_legal_fictions",
        "https://arxiv.org/pdf/2401.01301",
        "Dahl et al. (2024)",
        "Dahl, M., Magesh, V., Suzgun, M., & Ho, D.E. (2024). Large Legal Fictions: "
        "Profiling Legal Hallucinations in Large Language Models. "
        "Journal of Legal Analysis, 16(1), 64–93. https://doi.org/10.1093/jla/laae003"
    ),
    (
        "magesh_2024_hallucination_free",
        "https://arxiv.org/pdf/2405.20362",
        "Magesh et al. (2024)",
        "Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C.D., & Ho, D.E. (2024). "
        "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools. "
        "arXiv:2405.20362"
    ),
    (
        "zheng_2025_reasoning_focused_legal_retrieval",
        "https://arxiv.org/pdf/2505.03970",
        "Zheng et al. (2025)",
        "Zheng, L., Guha, N., Arifov, J., Zhang, S., Skreta, M., Manning, C.D., "
        "Henderson, P., & Ho, D.E. (2025). A Reasoning-Focused Legal Retrieval Benchmark. "
        "CS&Law '25. https://doi.org/10.1145/3709025.3712219"
    ),
    (
        "towards_robust_legal_reasoning_2025",
        "https://arxiv.org/pdf/2502.17638",
        "arXiv:2502.17638 (2025)",
        "Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law. arXiv:2502.17638"
    ),

    # ── 2. RAG Faithfulness — Mechanism ───────────────────────────────────────

    (
        "correctness_not_faithfulness_2024",
        "https://arxiv.org/pdf/2412.18004",
        "arXiv:2412.18004 (2024)",
        "Correctness is not Faithfulness in RAG Attributions. arXiv:2412.18004"
    ),

    # ── 3. Generation-Stage Faithfulness Interventions ────────────────────────

    (
        "asai_2023_selfrag",
        "https://arxiv.org/pdf/2310.11511",
        "Asai et al. (2023)",
        "Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). "
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. "
        "arXiv:2310.11511"
    ),
    (
        "ssfo_2025_self_supervised_faithfulness",
        "https://arxiv.org/pdf/2508.17225",
        "arXiv:2508.17225 (2025)",
        "SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation. "
        "arXiv:2508.17225"
    ),
    (
        "raglens_2025_sparse_autoencoders",
        "https://arxiv.org/pdf/2512.08892",
        "Xiong et al. (2025)",
        "Xiong, G. et al. (2025). Toward Faithful Retrieval-Augmented Generation with "
        "Sparse Autoencoders. arXiv:2512.08892"
    ),
    (
        "tamber_2025_faithjudge",
        "https://arxiv.org/pdf/2505.04847",
        "Tamber et al. (2025)",
        "Tamber, M.S. et al. (2025). Benchmarking LLM Faithfulness in RAG with "
        "Evolving Leaderboards. arXiv:2505.04847"
    ),
    (
        "ayala_bechard_2024_structured_rag",
        "https://arxiv.org/pdf/2404.08189",
        "Ayala & Bechard (2024)",
        "Ayala, O., & Bechard, P. (2024). Reducing hallucination in structured outputs "
        "via Retrieval-Augmented Generation. NAACL 2024 Industry Track."
    ),

    # ── 4. Legal-Specific Generation Interventions ────────────────────────────

    (
        "servantez_2024_chain_of_logic",
        "https://arxiv.org/pdf/2402.10400",
        "Servantez et al. (2024)",
        "Servantez, S., Barrow, J., Hammond, K., & Jain, R. (2024). "
        "Chain of Logic: Rule-Based Reasoning with Large Language Models. "
        "ACL Findings 2024. https://doi.org/10.18653/v1/2024.findings-acl.159"
    ),
    (
        "investigating_shortcomings_llm_legal_reasoning_2025",
        "https://arxiv.org/pdf/2502.05675",
        "arXiv:2502.05675 (2025)",
        "Investigating the Shortcomings of LLMs in Step-by-Step Legal Reasoning. "
        "arXiv:2502.05675"
    ),

    # ── 5. Calibration ────────────────────────────────────────────────────────

    (
        "guo_2017_calibration_neural_networks",
        "https://arxiv.org/pdf/1706.04599",
        "Guo et al. (2017)",
        "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). "
        "On Calibration of Modern Neural Networks. ICML 2017."
    ),

    # ── 6. General Reasoning / Prompting Methods ──────────────────────────────

    (
        "wei_2022_chain_of_thought",
        "https://arxiv.org/pdf/2201.11903",
        "Wei et al. (2022)",
        "Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., "
        "Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in "
        "Large Language Models. NeurIPS 2022."
    ),
    (
        "wang_2023_self_consistency",
        "https://arxiv.org/pdf/2203.11171",
        "Wang et al. (2023)",
        "Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., "
        "& Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in "
        "Language Models. ICLR 2023."
    ),
    (
        "bright_benchmark_2025",
        "https://arxiv.org/pdf/2407.12883",
        "BRIGHT (2025)",
        "Su, H. et al. (2025). BRIGHT: A Realistic and Challenging Benchmark for "
        "Reasoning-Intensive Retrieval. ICLR 2025."
    ),

    # ── 7. Query Expansion (Contrast Literature) ──────────────────────────────

    (
        "wang_2023_query2doc",
        "https://arxiv.org/pdf/2303.07678",
        "Wang et al. (2023)",
        "Wang, L., Yang, N., & Wei, F. (2023). Query2doc: Query Expansion with Large "
        "Language Models. EMNLP 2023."
    ),
    (
        "gao_2022_hyde",
        "https://arxiv.org/pdf/2212.10496",
        "Gao et al. (2022)",
        "Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise Zero-Shot Dense "
        "Retrieval without Relevance Labels (HyDE). arXiv:2212.10496"
    ),

    # ── 8. Legal Reasoning Taxonomy ───────────────────────────────────────────

    (
        "guha_2023_legalbench",
        "https://arxiv.org/pdf/2308.11462",
        "Guha et al. (2023)",
        "Guha, N. et al. (2023). LegalBench: A Collaboratively Built Benchmark for "
        "Measuring Legal Reasoning in Large Language Models. NeurIPS 2023."
    ),
    (
        "huhn_2003_stages_legal_reasoning",
        "https://digitalcommons.law.villanova.edu/cgi/viewcontent.cgi?article=1324&context=vlr",
        "Huhn (2003)",
        "Huhn, W.R. (2003). The Stages of Legal Reasoning: Formalism, Analogy, and "
        "Realism. Villanova Law Review, 48, 305."
    ),

    # ── 9. Foundational RAG / LLM Papers ─────────────────────────────────────

    (
        "lewis_2020_rag_original",
        "https://arxiv.org/pdf/2005.11401",
        "Lewis et al. (2020)",
        "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., "
        "Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). "
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020."
    ),

    # ── 10. Hallucination Surveys ─────────────────────────────────────────────

    (
        "hallucination_mitigation_rag_survey_2025",
        "https://arxiv.org/pdf/2510.24476",
        "arXiv:2510.24476 (2025)",
        "Mitigating Hallucination in Large Language Models: An Application-Oriented "
        "Survey on RAG, Reasoning, and Agentic Systems. arXiv:2510.24476"
    ),
    (
        "rag_comprehensive_survey_2025",
        "https://arxiv.org/pdf/2506.00054",
        "arXiv:2506.00054 (2025)",
        "Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, "
        "Enhancements, and Robustness Frontiers. arXiv:2506.00054"
    ),

    # ── 11. Additional Legal LLM Papers ───────────────────────────────────────

    (
        "lexam_2025_law_exams_benchmark",
        "https://arxiv.org/pdf/2505.12864",
        "LEXam (2025)",
        "LEXam: Benchmarking Legal Reasoning on 340 Law Exams. arXiv:2505.12864"
    ),
    (
        "llms_legal_reasoning_unified_framework_2025",
        "https://arxiv.org/pdf/2507.07748",
        "arXiv:2507.07748 (2025)",
        "When Large Language Models Meet Law: Dual-Lens Taxonomy, Technical Advances, "
        "and Ethical Governance. arXiv:2507.07748"
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOADER
# ──────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-paper-downloader/1.0; "
        "thesis research use)"
    )
}


def download_pdf(url: str, dest_path: Path, retries: int = 3) -> bool:
    """Download a PDF from url to dest_path. Returns True on success."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                # Accept PDF or octet-stream
                if "pdf" in content_type or "octet" in content_type or len(response.content) > 1000:
                    with open(dest_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    # Verify it looks like a PDF
                    with open(dest_path, "rb") as f:
                        header = f.read(5)
                    if header == b"%PDF-":
                        return True
                    else:
                        dest_path.unlink(missing_ok=True)
                        return False
            elif response.status_code == 404:
                return False
            else:
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Download literature review papers.")
    parser.add_argument("--output", default="papers", help="Output folder (default: ./papers)")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs only, no download")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between requests (default: 2.0, be polite to servers)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"Literature Review Paper Downloader")
    print(f"{'='*65}")
    print(f"Output folder : {out_dir.resolve()}")
    print(f"Total papers  : {len(PAPERS)}")
    print(f"Mode          : {'DRY RUN — no files written' if args.dry_run else 'DOWNLOAD'}")
    print(f"{'='*65}\n")

    if args.dry_run:
        for i, (stem, url, key, ref) in enumerate(PAPERS, 1):
            print(f"{i:02d}. [{key}]")
            print(f"     File : {stem}.pdf")
            print(f"     URL  : {url}")
            print()
        return

    results = {"success": [], "failed": [], "skipped": []}

    for stem, url, key, ref in tqdm(PAPERS, desc="Downloading"):
        dest = out_dir / f"{stem}.pdf"

        if dest.exists() and dest.stat().st_size > 10_000:
            tqdm.write(f"  ✓ SKIP (exists)  {stem}.pdf")
            results["skipped"].append(stem)
            continue

        tqdm.write(f"  ↓ {key}  →  {stem}.pdf")
        success = download_pdf(url, dest)

        if success:
            size_kb = dest.stat().st_size // 1024
            tqdm.write(f"    ✓ OK ({size_kb} KB)")
            results["success"].append(stem)
        else:
            tqdm.write(f"    ✗ FAILED — {url}")
            results["failed"].append((stem, url))

        time.sleep(args.delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*65}")
    print(f"  Downloaded : {len(results['success'])}")
    print(f"  Skipped    : {len(results['skipped'])} (already existed)")
    print(f"  Failed     : {len(results['failed'])}")

    if results["failed"]:
        print(f"\n  FAILED PAPERS (manual download needed):")
        for stem, url in results["failed"]:
            # Find citation
            cite = next((k for s, _, k, _ in PAPERS if s == stem), stem)
            print(f"    [{cite}]  {url}")
        print()

    # ── Write reference list ──────────────────────────────────────────────────
    ref_path = out_dir / "REFERENCES.txt"
    with open(ref_path, "w") as f:
        f.write("LITERATURE REVIEW — FULL REFERENCE LIST\n")
        f.write("=" * 65 + "\n\n")
        for i, (stem, url, key, ref) in enumerate(PAPERS, 1):
            f.write(f"[{i:02d}] {key}\n")
            f.write(f"     {ref}\n")
            f.write(f"     PDF: {stem}.pdf\n\n")
    print(f"\n  Reference list saved to: {ref_path}")
    print(f"  Papers folder         : {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
