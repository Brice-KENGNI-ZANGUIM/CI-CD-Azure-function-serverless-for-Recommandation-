######################################################################
########      Importation des bibliothèquees necessaires      ########
######################################################################

import logging
import azure.functions as func
import pandas as pd

from model_prediction import *

########################################################################################
############                    lecture des DataFrames                    ##############
########################################################################################

data = pd.read_csv("DataFrames/data.csv")
articles_metadata = pd.read_csv("DataFrames/articles_metadata.csv")
article_embedding_80 = pd.read_csv("DataFrames/article_embedding_80.csv")
article_embedding_80 = article_embedding_80.set_index( articles_metadata.article_id.values )


# Function trigger http request
def main (req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    #########################################################################
    #########       récupération des paramètres de la requête      ##########
    #########################################################################
    # user_id
    user_id = req.params.get('user_id')
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('user_id')
            
    # models
    models =  req.params.get('models')
    if not models:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            models = req_body.get('models')
            
    # recommand_count        
    n =  req.params.get('recommand_count')
    if not n:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            n = req_body.get('recommand_count')
            
    #        
    if_recommand = req.params.get('if_recommand')
    if not if_recommand:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            if_recommand = req_body.get('if_recommand')

    #########################################################
    #########       Recommandation d'articles      ##########
    #########################################################
    if if_recommand :
        # acquisition du DataFrame
        recommandations = prediction(models, data, articles_metadata, article_embedding_80, user_id,  n)

        # Transformation en chaine de caractères
        recommandations = recommandations.article_id.apply(str).values
        recommandations = "_".join(recommandations)

        return func.HttpResponse( recommandations )

    ##########################################################
    #######     Articles déjà conne de l'utilisateur    ######
    ##########################################################
    else :
        user_articles_list = anciens_articles_utilisateurs( data, articles_metadata, user_id )
        user_articles_list = user_articles_list.article_id.apply(str).values
        user_articles_list = "_".join(user_articles_list) 

        return func.HttpResponse( user_articles_list )

