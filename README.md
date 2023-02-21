# catoni-confseq
Confidence sequences (CSs) based on M-estimation for heavy-tailed, possibly contaminated data.

This repository serves as the code base for the following two papers, both authored by [Hongjian Wang](https://wanghongjian.wordpress.com/) and [Aaditya Ramdas](https://www.stat.cmu.edu/~aramdas/index.html):


- [Catoni-Style Confidence Sequences for Heavy-Tailed Mean Estimation](https://arxiv.org/abs/2202.01250)
- [Huber-robust Confidence Sequences](https://arxiv.org/abs/2301.09573)  *AISTATS 2023 (oral)*


The class `RCS_generator` in [`robustconfseq.py`](https://github.com/ShimonTroiaeAbOrisWang/catoni-confseq/blob/main/robustconfseq.py) enables online, sequential, robust estimation (uncertainty quantification) and hypothesis testing of the population mean. One initializes the class by supplying the noise rate `eps` (can be 0), an upper bound on the variance `moment` (or any other $p$-central moment with $p>1$), and an optional `null` hypothesis. *Whenever* a new datapoint arrives, evoke `observe` to record it. 

In the process, one may query the e-value, p-value, and the confidence intervals *ad libitum*, without worries about the traditionally unsafe behavior of continuous monitoring.

The confidence sequences are proven in both papers minimax optimal.

Related repos:
-[confseq](https://github.com/gostevehoward/confseq): contains CSs for light-tailed data.
-[confseq_wor](https://github.com/WannabeSmith/confseq_wor) contains CSs for sampling without replacement.
