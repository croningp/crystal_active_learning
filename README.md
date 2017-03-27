## Human vs Robots in the Discovery and Crystallization of Gigantic Polyoxometalates
Vasilios Duros+, Jonathan Grizou+, Weimin Xuan, Zied Hosni, De-Liang Long, Haralampos N. Miras and Leroy Cronin*

**Abstract:** Molecular discovery by crystallization is a challenging endeavor as it combines two serendipitous events; first is the formation of a new molecule, and second its crystallization. Herein, we constructed a workflow to probe the envelope of both events in the chemical space of a new polyoxometalate cluster, namely Na6[Mo120Ce6O366H12(H2O)78]·200H2O (1) (yield 4.3% based on Mo). This compound features a dodecameric ring with an inner diameter of 17 Å and an outer of 31 Å. We developed an active machine-learning algorithm to explore the crystallization space and compared it to human experimenters. The algorithm increased crystallization prediction accuracy to 82% over 77% from human experimenters. We also report the ability of our method to explore and discover new areas of interest. It is shown that different human experimenters use different exploration strategies according to their perceptions of the system and personal biases.


### Repository organization


- [analysis](analysis) contains all the script for the analysis of both explored space and learning progress, it generates most plots used in the SI
- [figures](figures) holds the script to generate the final figures in the paper
- [real_experiments](real_experiments) holds the data from the real experiments
- [simulation](simulation) folder contains the simulation script used to generate the results in the SI
- [utils](utils) folder includes the various mathematical and procedural tools
