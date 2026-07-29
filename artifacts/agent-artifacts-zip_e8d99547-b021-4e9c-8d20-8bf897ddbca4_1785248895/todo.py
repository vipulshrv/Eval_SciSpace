Planner Code plan:
```python
def main():
    # 1. Search for scholarly literature on AI-based early cancer detection methods.
    # The search query targets imaging, genomics, and multimodal approaches with a focus on metrics.
    search_query = (
        "AI-based early cancer detection methods: comparative study of imaging, genomics, "
        "and multimodal approaches focusing on AUC, sensitivity, and specificity performance metrics"
    )
    search_results = search_scholarly_literature(
        queries=[search_query],
        search_providers=["scispace", "google_scholar", "pubmed"]
    )

    # 2. Generate a comprehensive structured comparative report.
    # The write_report tool will extract the necessary metrics (AUC, sensitivity, specificity) 
    # and insights from the search results to create the final document.
    report_prompt = (
        "Create a structured comparative report on AI-based early cancer detection. "
        "The report must compare Imaging, Genomics, and Multimodal approaches. "
        "Include a detailed comparison of performance metrics: AUC, Sensitivity, and Specificity. "
        "Structure the report with an executive summary, sections for each approach, "
        "a comparative analysis table/section, and a conclusion with references."
    )
    write_report(
        user_prompt=report_prompt,
        source_context=search_results,
        output_format="markdown"
    )

if __name__ == "__main__":
    main()

```