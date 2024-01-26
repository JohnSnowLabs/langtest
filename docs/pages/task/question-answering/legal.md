
<div class="h3-box" markdown="1">

## Legal

The primary goal of the Legal Test is to assess a model’s capacity to determine the case (legal conclusion) that supports a given legal claim presented in a text passage. This evaluation aims to gauge the model’s proficiency in legal reasoning and comprehension.

**How it works:**

{:.table3}
| test_type      | case  | legal_claim                                                                                                                                                                                                                                                    | legal_conclusion_A                                                             | legal_conclusion_B                                                                                          | correct_conlusion                                                             | model_conclusion | pass |
|----------------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|------------------|------|
| legal-support  | See United States v. Franik, 687 F.3d 988, 990 (8th Cir.2012) (where defendant does not raise procedural error, court bypasses review and only reviews substantive reasonableness of sentence for abuse of discretion);.......| Upon careful review, we conclude that the district court did not abuse its discretion in sentencing Trice. | where defendant does not raise procedural error, court bypasses review and only reviews substantive reasonableness of sentence for abuse of discretion | where district court varied downward from Guidelines range, it was ""nearly inconceivable"" that court abused its discretion in not varying downward further | a                 | a                |<span style="color:green">True</span>|


#### Evaluation Criteria

-  In the evaluation we doing the string comparison between *correct_conlusion* and *model_conclusion*, Based on that we are calculating the pass.

</div>