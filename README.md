# 1. Reward Model相关
+ reward_agent/：Reward Model相关数据、训练、评估、生成数据
+ reward_agent/data/：用于训练reward model的数据
+ reward_agent/reward_model/：训练好保存的reward model
+ reward_agent/generateData_reward_agent.py：利用训练好的reward model生成对局数据

# 2. 基于Reward Agent生成的对局数据相关
+ reward_agent_data/：Reward Agent生成的对局数据
+ reward_agent_data/*/hands_category/：牌型预测相关数据集
+ reward_agent_data/nolimit_2player/hands_category/balanced_poker_hands.json：牌型预测平衡类别数据集

# 3. 牌型预测任务数据集构造相关
+ poker_hands2category_datatrans.py：将Reward Agent生成的数据数据转换为牌型预测数据
+ poker_hands_balancedDataset.py：从牌型预测数据中构造平衡类别数据集

# 4. 牌型预测GPT相关
+ poker_hands_tokenizer.py：德扑数据集Tokenizer
+ poker_hands_GPT_train.py：训练GPT用于德扑牌型预测
+ poker_hands_GPT_continue_train.py：加载已保存GPT模型进行Continue Training
+ poker_hands_GPT_eval.py：评估已保存的GPT模型在牌型预测任务中的表现