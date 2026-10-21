## Introduction 

● Opening hook
● Why explanations are needed
● Problem with explanation filters in this context
● Our research question
● List of Experiments


Explanations are needed to for accountability, to see where problems stem from in these black box systems and to better understand how they work 

Research question is 


### placeholder
Large language models learn only from text that survives data filtering, so these filters decide whose language is represented. Documented failures include C4’s blocklist removing LGBTQ+ and African American English text [cite] and toxicity classifiers flagging African American English as toxic [cite]. Such filters are audited only in aggregate, and none explains an individual removal. Post-hoc feature attribution methods such as SHAP and LIME could fill this gap by showing which words drive a removal. But an explanation helps an audit only if it is faithful, meaning it reflects what the filter actually did. Prior work often checks whether explainers agree with each other, which does not establish correctness. We ask: are SHAP and LIME faithful enough to support identity-term bias audits of a pretraining filter? We test them on Detoxify against two references where the correct explanation can be computed independently, namely counterfactual identity-word swaps and exact Shapley values from a transparent filter, and we add a deletion test and a LIME stability analysis. Our experiments suggest attribution can triage a filter for bias but cannot certify it as fair.
