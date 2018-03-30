import sys,pprint
import os
import json
import watson_developer_cloud
import quandl
import pandas as pd
import matplotlib.pyplot as plt
import config

pp = pprint.PrettyPrinter(indent=4)
watson_iterations= 1
watson_count = 50
financial_file_name = "sterling_to_euro.csv"
watson_hosts_file_name = "watson_hosts.json"
watson_general_file_name = "watson_general.json"
watson_pretty_hosts_file_name = "watson_hosts_pretty.json"
watson_pretty_general_file_name = "watson_general_pretty.json"

hosts = ['www.theguardian.com','www.bloomberg.com']
sys.path.append(os.path.join(os.getcwd(),'..'))
quandl.ApiConfig.api_key = config.QUANDL_KEY

discovery = watson_developer_cloud.DiscoveryV1(
    username=config.WATSON_USERNAME,
    password=config.WATSON_PASSWORD,
    version=config.WATSON_VERSION
)


def get_financial_data():
    # Get data for past 3 months
    data = quandl.get("BOE/XUDLSER", start_date="2017-2-2", end_date="2017-4-5")
    # print("Financial Data:")
    # print(type(data))
    # print(data)
    return (data)


def write_financial_data(file_name, data):
    # with takes care of closing file in python
    with open(file_name, 'w') as outfile:
        data.to_csv(outfile)
    outfile.close()


def write_watson_data(file_name, data):
    with open(file_name, 'w') as outfile:
        outfile.write(json.dumps(data, outfile))
    outfile.close()


def write_watson_data_pretty(file_name, data):
    with open(file_name, 'w') as outfile:
        # Use this to write dictionaries
        # seperate into responses returned from watson
        for response in data:
            # Each response from watson contains 50 results
            outfile.write(json.dumps(response, outfile))
            outfile.write('\n')
    # python closes file at end of function scope but no harm in being explicit
    outfile.close()


def get_watson_data(query, filter, watson_index, watson_count):
    # get environment for this service instance
    environments = discovery.get_environments()
    # pp.pprint(environments)

    # identify the news environment using list comprehensions
    news_environments = [x for x in environments['environments'] if
                         x['name'] == 'Watson News Environment']
    # parse JSON for news environment id
    news_environment_id = news_environments[0]['environment_id']
    # pp.pprint(news_environment_id)

    # get news collection from news environment
    collections = discovery.list_collections(news_environment_id)
    news_collections = [x for x in collections['collections']]
    news_collection_id = news_collections[0]['collection_id']
    # pp.pprint(news_collection_id)
    # pp.pprint(collections)

    # Set query filters and get response
    qopts = {
        'query': query,
        'count': str(watson_count),
        'offset': str(watson_count * watson_index),
        'return': filter
    }
    watson_response = discovery.query(news_environment_id, news_collection_id, qopts)
    print(type(watson_response))
    # print(json.dumps(watson_response, indent=2))
    return watson_response


def onclick(event, df):
    index = event.ind
    print('onclick index:', index)
    for textList in df.iloc[index]['snippet']:
        for text in textList:
            print(text, sep='\n')

        # Helper function to replace list of nested dictionary with value
        def explode_dict(entity_list):
            for analysed_text in entity_list:
                if 'sentiment' in analysed_text and 'score' in analysed_text['sentiment'] and analysed_text[
                    'text'] == "Brexit":
                    return analysed_text['sentiment']['score']
                else:
                    return None


def create_watson_query(use_hosts):
    watson_data = []
    if use_hosts:
        for host in hosts:
            for watson_index in range(watson_iterations):
                # fetch data from API
                query = 'entities.text:brexit,pound,publicationDate.confident:yes,blekko.host:' + host
                filter = 'publicationDate,entities.sentiment,entities.text,blekko.snippet,blekko.host',
                watson_response = (get_watson_data(query, filter, watson_index, watson_count))

                # add results to total response
                watson_entities = watson_response['results']
                watson_data.extend(watson_entities)
        return watson_data
    else:
        for watson_index in range(watson_iterations):
            # fetch data from API
            query = 'entities.text:brexit,pound,publicationDate.confident:yes'
            filter = 'publicationDate,entities.sentiment,entities.text',
            watson_response = (get_watson_data(query, filter, watson_index, watson_count))

            # add results to total response
            watson_entities = watson_response['results']
            watson_data.extend(watson_entities)
        return watson_data


# Helper function to replace list of nested dictionary with value
def reduce_dict_list(entity_list):
    print("entity List:")
    print(entity_list)
    for analysed_text in entity_list:
        if 'sentiment' in analysed_text and 'score' in analysed_text['sentiment'] and analysed_text['text'] == "Brexit":
            return analysed_text['sentiment']['score']
        else:
            return None



# get financial data and write to csv
financial_data = get_financial_data()
watson_host_data = create_watson_query(use_hosts = True)
watson_general_data = create_watson_query(use_hosts = False)
print(watson_general_data)

# Write Quandle exchange rate data to CSV file
write_financial_data(financial_file_name, financial_data)

# Write watson sentiment analysis to json files
write_watson_data(watson_hosts_file_name, watson_host_data)
write_watson_data(watson_general_file_name, watson_general_data)

# write watson data in readable format
write_watson_data_pretty(watson_pretty_hosts_file_name, watson_host_data)
write_watson_data_pretty(watson_pretty_general_file_name, watson_host_data)

# with takes care of closing file after reading
with open(watson_hosts_file_name) as data_file:
    watson_hosts_data = json.load(data_file)

with open(watson_general_file_name) as data_file:
    watson_general_data = json.load(data_file)

with open(financial_file_name) as data_file:
    financial_data = pd.read_csv(data_file)

watson_dfs = []
watson_dfs.append(pd.DataFrame(watson_hosts_data))
watson_dfs.append(pd.DataFrame(watson_general_data))

financial_df = pd.DataFrame(financial_data)

for (df_index, watson_df) in enumerate(watson_dfs):
    watson_df.drop(['id', 'score'], axis=1, inplace=True)
    # Extract values from dictionaries
    watson_df = pd.concat(
        [watson_df.drop('publicationDate', axis=1), pd.DataFrame(watson_df['publicationDate'].tolist())], axis=1)

    if df_index == 0:
        watson_df = pd.concat([watson_df.drop('blekko', axis=1), pd.DataFrame(watson_df['blekko'].tolist())], axis=1)

    # print("watson_df.head(5):")
    # print(watson_df.head(5))

    watson_df = pd.concat([watson_df.drop('entities', axis=1), watson_df['entities'].apply(reduce_dict_list)], axis=1)
    watson_df = pd.concat(
        [watson_df.drop('entities', axis=1), pd.DataFrame(watson_df['entities'].tolist(), columns=['score'])], axis=1)

    # print(watson_df.head(5))
    watson_df = watson_df[watson_df.score.notnull()]

    # print("watson_df after split:")
    # print(watson_df.head(5))

    # Format date column into dateTime structure
    watson_df['date'] = watson_df.date.str[:8]
    watson_df['date'] = pd.to_datetime(watson_df['date'], format='%Y%m%d')
    watson_df.set_index('date')
    watson_df['score'] = watson_df.score.astype(float)
    watson_dfs[df_index] = watson_df

financial_df['Date'] = pd.to_datetime(financial_df['Date'], format='%Y-%m-%d')
financial_df.set_index('Date')
print(financial_df)
# convert strings to floats


#directive so IPython displays plots in notebook cell
#%matplotlib inline
fig, axes = plt.subplots(nrows=1, ncols=2)

watson_dfs[0].plot(x='date', y='score', style='ro',ax=axes[0])
financial_data.plot(x="Date",ax=axes[1])

fig.subplots_adjust(wspace=0.5)

axes[0].set_title("Watson Sentiments")
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Sentiment Score')
axes[1].set_title("Sterling Value")
axes[1].set_ylabel('Sterling Value')

fig = plt.figure()
ax1 = fig.add_subplot(111)
# get scores for a given media provider
col = ax1.plot(watson_dfs[1]['score'], picker=True, marker='o', linestyle='--', color='r')
col = ax1.plot(watson_dfs[1]['score'], picker=True, marker='o', linestyle='--', color='b')

fig = plt.figure()
ax1 = fig.add_subplot(111)
# get scores for a given media provider
col = ax1.plot(watson_dfs[0].loc[watson_dfs[0]['host'] == 'www.bloomberg.com', 'score'], picker=True, marker='o', linestyle='--', color='r')
col = ax1.plot(watson_dfs[0].loc[watson_dfs[0]['host'] == 'www.theguardian.com', 'score'], picker=True, marker='o', linestyle='--', color='b')

fig.canvas.mpl_connect('pick_event', lambda event: onclick(event, watson_df))

plt.show()