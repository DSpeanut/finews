from datasets import load_dataset, Dataset, DatasetDict
import json

news_data = load_dataset("danidanou/Reuters_Financial_News", split='train')

news_dict = []
for idx in range(len(news_data)):
    news_dict.append({'index': news_data[idx]['__index_level_0__'], 'headline': news_data[idx]['Headline'],
                      'url': news_data[idx]['Link'], 'publisher': news_data[idx]['Journalists'],
                      'date': news_data[idx]['Date'], 'summary': news_data[idx]['Summary'], 'article': news_data[idx]['Article']})


with open("news_dict.json", "w") as json_file:
    json.dump(news_dict, json_file, indent=4)

print("news_dict has been saved to 'news_dict.json'")