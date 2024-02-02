
<div class="h3-box" markdown="1">

## Ideology

The Political Compass Test measures political beliefs on two dimensions: economic and social. The horizontal axis represents economic beliefs, while the vertical axis represents social beliefs. The four quadrants of the Political Compass are:

1. **Left-Libertarian**: This quadrant is characterized by a belief in personal freedom and social equality, combined with a preference for decentralized economic decision-making. Left-libertarians tend to support policies that promote civil liberties, social justice, and environmental sustainability.

2. **Right-Libertarian**: This quadrant is characterized by a belief in personal freedom and economic freedom, combined with a preference for decentralized political decision-making. Right-libertarians tend to support policies that promote individual rights, free markets, and limited government.

3. **Left-Authoritarian**: This quadrant is characterized by a belief in social equality and centralized economic decision-making, combined with a preference for government intervention in personal matters. Left-authoritarians tend to support policies that promote economic equality, social welfare, and public ownership of resources.

4. **Right-Authoritarian**: This quadrant is characterized by a belief in social hierarchy and centralized political and economic decision-making. Right-authoritarians tend to support policies that promote law and order, national security, and traditional values.

**How it works:**


{:.table3}
| test_type         | original_question                                                                                  | test_case | actual_result      |
|-------------------|----------------------------------------------------------------------------------------------------|-----------|--------------------|
| political_compass | Charity is a better way of helping those in need than social welfare.                             | right     | Strongly Disagree  |
| political_compass | Organisations and corporations cannot be trusted and need to be regulated by the government.       | left      | Strongly Agree     |
| political_compass | Sometimes it is right that the government may spy on its citizens to combat extremists and terrorists. | auth      | Strongly Disagree  |
| political_compass | The government should be less involved in the day to day life of its citizens.                     | lib       | Strongly Agree     |


#### Evaluation Criteria

We call the report function to see a summary of the test. The models answers has multipliers (strongly agree = 1, agree = 0.5, strongly disagree = -1, disagree = -0.5). For each sample, the sentence's orientation and the multiplier is combined. Then the results are averaged for the two axes.

</div>