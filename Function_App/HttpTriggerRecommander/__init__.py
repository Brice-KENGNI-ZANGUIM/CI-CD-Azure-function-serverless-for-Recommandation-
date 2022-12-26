######################################################################
########      Importation des bibliothèquees necessaires      ########
######################################################################

import logging
import azure.functions as func
from io import StringIO

##################################
##################################

import os
import pandas as pd
from .model_prediction import prediction, anciens_articles_utilisateurs
from .My_functions import blob_stream_to_dataframe_v2

########################################################################################
############                    lecture des DataFrames                    ##############
########################################################################################

#cwd = os.getcwd()

#data = pd.read_csv( "/".join( [cwd , "DataFrames/data.csv"] ) )
#articles_metadata = pd.read_csv( "/".join( [cwd ,"DataFrames/articles_metadata.csv"] ) )
#article_embedding_80 = pd.read_csv( "/".join( [cwd , "DataFrames/article_embedding_80.csv"] ) )

#article_embedding_80 = article_embedding_80.set_index( articles_metadata.article_id.values )

# Function trigger http request
def main (  req: func.HttpRequest,
            data : func.InputStream,
            articleembedding80 : func.InputStream,
            articlesmetadata : func.InputStream ,
            ) -> func.HttpResponse :
    logging.info("Le trigger s'est déclenché avec succès et les fichier ont bien étés envoyés à la fonction lambda")
	
    #########################################################################
    #########      Conversion des données Stream en DataFrame      ##########
    #########################################################################
    #return func.HttpResponse( data.read() )
    #return func.HttpResponse( f"data  {len(data.read( ))} - articleembedding80  {len(articleembedding80.read( ))} - articlesmetadata  {len(articlesmetadata.read( ))}" )
    articles_metadata = blob_stream_to_dataframe_v2( articlesmetadata )
    article_embedding_80 =  blob_stream_to_dataframe_v2( articleembedding80 )
    data = blob_stream_to_dataframe_v2( data )

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
    user_id = int(user_id)

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
    n = int(n)

    #  if_recommand      
    if_recommand = req.params.get('if_recommand')
    if not if_recommand:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            if_recommand = req_body.get('if_recommand')
    if_recommand = int(if_recommand)

    #return func.HttpResponse(f"{user_id}   -  {models}  -   {if_recommand} -  {n} ")
    if user_id or if_recommand or models or n : 
        #########################################################
        #########       Recommandation d'articles      ##########
        #########################################################
        if if_recommand  :

            # acquisition du DataFrame
            recommandations = prediction(models, data, articles_metadata, article_embedding_80, user_id,  n)

            # Transformation en chaine de caractères
            #recommandations = recommandations.article_id.apply(str).values
            #recommandations = "_".join(recommandations)

            return func.HttpResponse( recommandations.to_csv(index=False).encode() )

        ##########################################################
        #######     Articles déjà conne de l'utilisateur    ######
        ##########################################################
        else :

            user_articles_list = anciens_articles_utilisateurs( data, articles_metadata, user_id )
            #user_articles_list = user_articles_list.article_id.apply(str).values
            #user_articles_list = "_".join(user_articles_list) 

            return func.HttpResponse( user_articles_list.to_csv(index=False).encode() )
    else :
        return func.HttpResponse(
                                "Cette fonction s'exécute très bien. vous pouvez passez des paramètres en requête",
                                status_code=200
                                )
