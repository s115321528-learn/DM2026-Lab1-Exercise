**Name:** 莊財瑋  
**Student ID:** 115321528

In this lab, I tried using an AI Agent to complete a data mining task. This was a little different from my previous assignments because I didn't need to run every function by myself. Instead, I had to tell the Agent what I wanted to do and let it choose and execute the tools.

At first, I thought this would be easier than writing Python code directly. However, after actually trying it, I found that using an AI Agent also requires some understanding of the tools and the data. If my instructions were not clear enough, the Agent might ask more questions or return results that were not exactly what I expected.

I also learned that even when the Agent says a task was completed successfully, I still need to check whether the results make sense.

I used the 20 Newsgroups dataset, which contains 2,257 documents from four categories.

After loading the dataset (`load_dataset_1`), I checked for missing values and duplicate documents. Fortunately, there were no missing texts or duplicate documents.

Next, I used `build_dtm_tool` (`dtm_2`) to convert the documents into a Document-Term Matrix.

The matrix contained 35,788 features, which was much larger than I expected. I was also surprised that the sparsity was about 99.55%.

This helped me understand why text mining often needs feature selection and dimensionality reduction.

I also generated a DTM heatmap (`dtm_heatmap_3`) to see how the words were distributed across different documents.

![DTM Heatmap](../plots/莊財瑋_agentic-pipeline_plot_dtm_heatmap_3.png)

One thing that surprised me was the feature selection results.

I tried Variance Filtering (`variance_4`), Pearson Filtering (`pearson_5`), and Spearman Filtering (`spearman_6`).

All three tools ran successfully, but none of them removed any features because the default threshold was 0.

At first, I thought something was wrong with my tools. Later, I realized that the threshold was the main reason.

This made me understand that just running a tool successfully doesn't mean I have achieved the purpose of the analysis. The parameters are also important.

I also generated a correlation matrix (`feature_corr_7`) and a bar chart using `pearson_5`.

![Feature Correlation Matrix](../plots/莊財瑋_agentic-pipeline_plot_feature_corr_7.png)

![Pearson Result](../plots/莊財瑋_agentic-pipeline_plot_turn50.png)

Another interesting part was PCA (`reduce_dim_pca_8`).

I reduced the original 35,788 features to only two dimensions.

![PCA Result](../plots/莊財瑋_agentic-pipeline_plot_reduce_dim_pca_8.png)

I originally expected the four categories to form four separate groups. However, most of the points overlapped.

There were also a few points far away from the others.

This result was different from what I imagined, but it helped me understand that PCA doesn't always separate different categories clearly.

Finally, I used `cosine_similarity_tool` (`cosine_sim_10`) to compare two documents from the same category.

The similarity score was only **0.353241**.

I thought documents from the same category would have higher similarity, so this result was quite interesting to me.

It showed me that documents can belong to the same category but still discuss very different topics.


I encountered several problems during this lab.

The first problem was the Gemini API. While I was using the Agent, I received a `402 RESOURCE_EXHAUSTED` error because the API credits were depleted.

At first, I wasn't sure whether the problem came from my code or the API. After checking the error message, I realized it was an API billing issue.

Another problem happened when I restarted JupyterLab.

I found that the previous Session results were gone, even though my conversation logs and plots were still saved.

Because of this, I had to load the dataset and build the DTM again before continuing.

This was probably one of the most frustrating parts of the experiment because I had already completed several steps.

Fortunately, I had backed up my files, so I didn't need to start everything from the beginning.

I also spent some time understanding the tool permission system.

The Agent asked me to approve access to the student-built tools. After approving the request, all 18 tools became available.

At first, this process felt a little complicated, but I later understood why an Agent should ask for permission before executing tools.


After finishing this lab, I feel that AI Agents are useful, but they are not as simple as just asking a question and waiting for an answer.

I still needed to know what I wanted to analyze, choose suitable tools, and check the results.

The part I liked most was being able to complete different data mining tasks through a conversation. It felt more interactive than running Python functions one by one.

However, I also found that debugging an Agent system can sometimes be more difficult because problems may come from the API, the tools, the session, or the instructions.

Through this experiment, I became more familiar with DTM, feature selection, PCA, frequent pattern mining, and cosine similarity.

More importantly, I learned that **AI can help me perform an analysis, but understanding and checking the results is still my responsibility.**

Overall, I think this was an interesting experiment. Although I encountered several problems, solving them helped me understand how an AI Agent works in a real data mining workflow.