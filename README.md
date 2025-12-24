# Fake-news-prediction using Machine Learning.
**Author**  : Rizan Atlanta (Fariha Hossain Rizan).<br/>
**Project** : Data science / Machine Learning.<br/>
**Status**  : Completed (Temporarily).


## Project Overview
This project presents a machine learning-based approach to detecting fake news by classifying news articles as **Fake** or **Credible**. This objective is not only to build a classifier but to **systematically evaluate** and compare multiple models, interpret results, and understand the limitation of text-based fake news detection.<br/>
The project follows a complete data science workflow including preprocessing, feature extraction, model training, evaluation, and model comparison.

## Dataset Description
The dataset consists of labeled news article with the following classes:
- Class 0: Fake news.
- class 1: Credible news.
- Link   : https://drive.google.com/file/d/1mdGAY0FllXEkYdHebvgNFlazAGWzZR56/view?usp=sharing

Each sample contains textual content used for classification. Due to the subjective and contextual nature of fake news, the dataset contains linguistic ambiguity and noise, making classification a **challenging task.**

## Data Preprocessing 
The following preprocessing steps were applied to prepare the text data for modelling:
- Conversion to lowercase.
- Removal of punctuation and special characters.
- Stopword removal.
- Tokenization.
- Feature extraction using TF-IDF.

These steps help reduce noise while preserving meaningful textual patterns.

## Feature Engineering
Textual features were transformed into numerical representation using TF-IDF, which assigns higher importance to informative words while inducing of frequently occuring but less meaningful terms.

## Model Training
Multiple machine learning models were trained to evaluate their effectiveness on high-dimensional text data:
- Decision Tree Classifier.
- Logistic Regression.
- Support Vector Machine (SVM) (with hyperparameter tuning)
The use of multiple models allows for a fair conmparison  and prevents reliance on a single approach.

## Model comparison
|**Model**                 | **Accuracy** | **F1-score (Class 0-Fake)** | **F1-score (Class 1-Credible)** |
|--------------------------|--------------|-----------------------------|---------------------------------|
| Decision Tree Classifier |  *48.8%*     |  *o.48*                     |  *0.50*                         |
| Logistic Regression model|  *52.7*      |  *0.40*                     |  *0.61*                         |
| Tuned SVM Model          |  *52.78*     |  *0.43*                     |  *0.60*                         |

## Analysis
Logistic Regression and the tuned SVM achieved the highest overall accuracy (52.7%). Logistic Regression demonstrated the strongest perfomance in identifying credible news, achieving the highest F1-score for class 1.<br/>
The Decision Tree classifier showed comparatively weaker performance, suggesting limitations in generalizing over sparse, higher-dimensional text features. Overall, linear models provided more stable and interpretable results for this dataset.

## Final Model selection 
Logistic Model regression was selected as the final model due to its balanced performance, interpretability, and consistent behavior across evaluation metrics.

##Limitations and Future Improvements
Fake news detection is inherently complex due to language ambiguity, sarcasm, and lack of contextual information. As a result, Model performance remained moderate despite experimentation with multiple approaches. <br/>

Future improvements may include:
- Incorporating n-gram features.
- Using advanced word embeddings such as Word2Vec or BERT.
- Expanding the dataset to improve  generalization.
- Exploration deep learning-based text classification models.

## Demonstration 
A working demonstration of the system, along with screenshots and a demo video with sample predictions is available via google drive.
**Screenshot(s):**
1. Screenshot 1 : https://drive.google.com/file/d/1qTRRAA5MmdjK_x1I1L-JDg5-_d1HdAXU/view?usp=sharing
2. Screenshot 2 : https://drive.google.com/file/d/16ElRLTD_oWWINQL5uLmtuFLc_qwTy3ji/view?usp=sharing
3. Screenshot 3 : https://drive.google.com/file/d/121GJbT-7wqv20pg5R4kl5r4Jln-Rl9Oz/view?usp=sharing
**Demo video:**
https://drive.google.com/file/d/1L9ifIQFeaW0ir-w_FqUXaSlJ02WLKjFZ/view?usp=sharing

## Technology Used
- Python.
- Scikit-learn.
- Pandas.
- NumPy.
- TF-IDF Vectorizer.

## References 
1. Shu, K., A., Wang, S., Tang,J., &Liu, H. (2017). <br/>
*Fake news detection on social media: A data mining perspective. ACM SIGKDD Exploration,19(1), 22-36*
2. Wang, W.Y. (2017). *"Liar, Liar Pants on Fire": *A new benchmark datset for fake news detection. Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL), 422-426*
3. Devlin, J., Chang, M. W.,Lee,K.,& Toutanova,K. (2019). BERT : *Pre-Training of deep bidirectional transformers for language understanding. NAACL.*
4. Soveatin Kuntur, M., Krywda,M., Wroblewska,A., Paprzycki,M., & Ganzha, M. (2024). *Comparative analysis of graph neural networks and transformers for robust fake news detection.* Electronics. This study empirically evaluates transformers-based NLP models. (e.g. BERT,RoBERTa, GPT-2).
5. An Unsupervised Fake News Detection Framework Based on Structural Contrasive Learning (2025).
6. Pedrogosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ...& Duchesnay, E.(2011). *Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825-2830.*

## Conclusion
This project presents a complete pipeline for fake news detection, covering data preprocessing, feature extraction, model training, and evaluation. The final model achieved an accuracy of 52.7%, which reflects the inherent difficulty of the task rather than a lack of methodological rigor.<br/>
Overall, the project emphasizes sound methodology, transparent, evaluation, and realistic interpretation of results,forming a strong foundation improvements using more advanced models and richer representations.
