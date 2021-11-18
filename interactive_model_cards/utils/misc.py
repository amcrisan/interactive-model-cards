import pandas as pd
from numpy import floor


#--- gensim ---
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def conf_level(val):
    """ Translates probability value into
        a plain english statement """
    # https://www.dni.gov/files/documents/ICD/ICD%20203%20Analytic%20Standards.pdf
    conf = "undefined"

    if val < 0.05:
        conf = "Extremely Low Probability"
    elif val >= 0.05 and val < 0.20:
        conf = "Very Low Probability"
    elif val >= 0.20 and val < 0.45:
        conf = "Low Probability"
    elif val >= 0.45 and val < 0.55:
        conf = "Middling Probability"
    elif val >= 0.55 and val < 0.80:
        conf = "High  Probability"
    elif val >= 0.80 and val < 0.95:
        conf = "Very High Probability"
    elif val >= 0.95:
        conf = "Extremely High Probability"

    return conf


def subsample_df(df=None, size=10, sample_type="Random Sample"):
    """ Subsample the dataframe  """
    size = int(size)
    if sample_type == "Random Sample":
        return df.sample(size)
    elif sample_type == "Highest Probabilities":
        df.sort_values(by="probability", ascending=False, inplace=True)
        return df.head(size)
    elif sample_type == "Lowest Probabilities":
        df.sort_values(by="probability", ascending=True, inplace=True)
        return df.head(size)
    else:
        # sample probabilities in the middle
        tmp = df[(df["probability"] > 0.45) & (df["probability"] < 0.55)]
        samp = min([size, int(tmp.shape[0])])
        return tmp.sample(samp)


def down_samp(embedding):
    """Down sample a data frame for altiar visualization """
    #total number of positive and negative sentiments in the class
    total_size = embedding.groupby(['name', 'sentiment'],as_index=False).count()

    user_data = 0
    if 'Your Sentences' in str(total_size['name']):
        tmp = embedding.groupby(['name'],as_index=False).count()
        val = int(tmp[tmp['name'] == "Your Sentences"]['source'])
        user_data=val

    max_sample = total_size.groupby('name').max()['source']

    #down sample to meeting altair's max values
    #but keep the proportional representation of groups
    down_samp = 1/(sum(max_sample)/(5000-user_data))

    max_samp = floor(max_sample*down_samp).astype(int).to_dict()
    max_samp['Your Sentences'] = user_data

    #sample down for each group in the data frame
    embedding= embedding.groupby('name').apply(lambda x: x.sample(n=max_samp.get(x.name))).reset_index(drop = True)

    #order the embedding
    return(embedding.sort_values(['sort_order'],ascending=True))



def prep_embed_data(data,model):
    ''' Basic data tagging'''
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    embedding = [model.infer_vector(tagged_data[i].words) for i in range(len(tagged_data))]
    return embedding

def prep_sentence_embedding(name,source, sentence, sentiment, sort_order,embed_model,idx,type="single"):
    """ Prepare a custom sentence to add to the embedding"""
    
    
    if type == "single":
        #get vector embedding
        tagged_data = TaggedDocument(words=word_tokenize(sentence.lower()), tags=['source'])
        vector = embed_model.infer_vector(tagged_data.words)

        tmp = {
            'source': source,
            'name': name,
            'sort_order': sort_order,
            'sentence': sentence,
            'sentiment': sentiment,
            'x': vector[0],
            'y':vector[1]
        }

        return(pd.DataFrame(tmp,index=[idx]))
    else:
        #go through each group and add 
        df = {"source":[],
            "name":[],
            "sentence":[],
            "sentiment":[],
            "x":[],
            "y":[],
            "sort_order":[]
        }


        slice_short = sentence
        slice_sentiment = sentiment
        vec_embedding = prep_embed_data(sentence,embed_model)

        df['source'] = df['source'] + [source]*len(slice_short)
        df['name'] = df['name'] + [name]*len(slice_short)

        #the sort order effects how its drawn by altair
        df['sort_order'] = df['sort_order'] + [sort_order]*len(slice_short)

        #add individual elements
        for i in range(len(slice_short)):
            df['sentence'].append(slice_short[i])
            df['sentiment'].append(slice_sentiment[i])
            df['x'].append(vec_embedding[i][0])
            df['y'].append(vec_embedding[i][1])

        df = pd.DataFrame(df) 
        return(df)      


