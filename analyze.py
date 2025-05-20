import pandas as pd
import json
from datetime import datetime
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import ast


main_categories = {
    '电子产品': ['智能手机', '笔记本电脑', '平板电脑', '智能手表', '耳机', '音响', '相机', '摄像机', '游戏机'],
    '服装': ['上衣', '裤子', '裙子', '内衣', '鞋子', '帽子', '手套', '围巾', '外套'],
    '食品': ['零食', '饮料', '调味品', '米面', '水产', '肉类', '蛋奶', '水果', '蔬菜'],
    '家居': ['家具', '床上用品', '厨具', '卫浴用品'],
    '办公': ['文具', '办公用品'],
    '运动户外': ['健身器材', '户外装备'],
    '玩具': ['玩具', '模型', '益智玩具'],
    '母婴': ['婴儿用品', '儿童课外读物'],
    '汽车用品': ['车载电子', '汽车装饰']
}


category_mapping = {}
for main, subs in main_categories.items():
    for sub in subs:
        category_mapping[sub] = main


with open('./product_catalog.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)


id_to_main = {}
id_to_price = {}
for product in catalog['products']:
    sub_category = product['category']
    main_category = category_mapping.get(sub_category, '其他')
    id_to_main[product['id']] = main_category
    id_to_price[product['id']] = product['price']


import glob


file_list = sorted(
    glob.glob('./30G_data_new/part-*.parquet'),
    key=lambda x: int(x.split('-')[-1].split('.')[0])
)


df = pd.concat(
    (pd.read_parquet(file) for file in file_list),
    ignore_index=True
)


df['purchase_history'] = df['purchase_history'].apply(ast.literal_eval)
df['items'] = df['purchase_history'].apply(lambda x: x.get('items', []))
df['payment_method'] = df['purchase_history'].apply(lambda x: x.get('payment_method'))
df['payment_status'] = df['purchase_history'].apply(lambda x: x.get('payment_status'))
df['purchase_date'] = df['purchase_history'].apply(lambda x: x.get('purchase_date'))


df['categories'] = df['items'].apply(lambda items: list(set([id_to_main.get(item['id'], '其他') for item in items])))

sample_product_check = df['items'].iloc[0][0]['id']
print(f"\n示例商品ID {sample_product_check} 映射到类别：{id_to_main.get(sample_product_check)}")
print("\n类别分布统计：")
print(df.explode('categories')['categories'].value_counts(normalize=True))


def task1():

    transactions = df[df['categories'].apply(len) >= 1]['categories'].tolist()

    expanded_transactions = []
    for t in transactions:
        from itertools import combinations
        if len(t) >= 2:
            expanded_transactions.append(t)
        else:
            expanded_transactions.append(t)
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(expanded_transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    

    if '电子产品' in df_trans.columns:
        electronic_support = df_trans['电子产品'].mean()
        print(f"\n电子产品独立支持度: {electronic_support:.4f}")
        

        adjusted_support = max(0.02, electronic_support/2)
    else:
        adjusted_support = 0.02
    

    frequent_itemsets = apriori(
        df_trans, 
        min_support=adjusted_support, 
        use_colnames=True,
        max_len=3 
    )
    

    if not frequent_itemsets.empty:
        rules = association_rules(
            frequent_itemsets, 
            metric='confidence', 
            min_threshold=0.5
        ).sort_values('lift', ascending=False)
        
        electronics_rules = rules[
            (rules['lift'] > 1) & 
            (rules.apply(lambda x: 
                ('电子产品' in x['antecedents']) or 
                ('电子产品' in x['consequents']), axis=1))
        ]
        
        if not electronics_rules.empty:
            electronics_rules['antecedents'] = electronics_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            electronics_rules['consequents'] = electronics_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            print("\n任务1结果（电子类相关规则）：")
            print(electronics_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        else:
            print("\n警告：即使调整参数后仍未发现符合条件的电子类关联规则")
    else:
        print("\n警告：未生成任何频繁项集，请检查数据或降低支持度阈值")


def task2():

    transactions_payment = []
    for _, row in df.iterrows():
        payment = f"支付_{row['payment_method']}"
        transaction = [payment] + row['categories']
        transactions_payment.append(transaction)
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions_payment)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
    
    print("\n任务2结果（支付方式关联规则）：")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']].head())
    
    # 高价值商品分析
    df['high_value'] = df['items'].apply(
        lambda items: any(id_to_price.get(item['id'], 0) > 5000 for item in items))
    high_value_payment = df[df['high_value']]['payment_method'].value_counts(normalize=True)
    
    print("\n高价值商品支付方式分布：")
    print(high_value_payment)

def task3():

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['month'] = df['purchase_date'].dt.month
    df['quarter'] = df['purchase_date'].dt.quarter
    

    monthly_counts = df.explode('categories').groupby(['month', 'categories']).size().unstack()
    quarterly_counts = df.explode('categories').groupby(['quarter', 'categories']).size().unstack()
    
    print("\n任务3结果（月度购买分布示例）：")
    print(monthly_counts[['电子产品', '服装']].head())
    

def task4():
    refund_df = df[df['payment_status'].isin(['已退款', '部分退款'])]
    transactions = refund_df['categories'].tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_trans, min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.4)
    
    print("\n任务4结果（退款关联规则）：")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']].head())


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()