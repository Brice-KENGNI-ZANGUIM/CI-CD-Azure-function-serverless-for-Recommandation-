########################################################################################
############           Importation de bibliothèques dépendantes           ##############
########################################################################################

import streamlit as st


########################################################################################
############     Importation de mes fonctions et classes personnelles     ##############
########################################################################################
from My_functions import *

########################################################################################
##########      Articles déjà consultés par l'utilisateur        #######################
########################################################################################
@st.cache(suppress_st_warning=True)
def anciens_articles_utilisateurs( data, articles_metadata, user_id ) :

    _data = data.copy()
    _articles_metadata = articles_metadata.copy()

    ##################################################################
    ##########   Articles déjà consultés par l'utilisateur  ##########
    ##################################################################

    return _articles_metadata.loc[ _articles_metadata.article_id.apply( 
                                                                        lambda x : x in _data.loc[ _data.user_id == user_id].article_id.values
                                                                        ).values ].reset_index(drop=True)[["article_id"]]


########################################################################################
################        Recommandations des articles        ############################
########################################################################################
@st.cache(suppress_st_warning=True)
def prediction(models, data, articles_metadata, article_embedding_80, user_id, n) :

    _models = models
    _data = data.copy()
    _articles_metadata = articles_metadata.copy()
    _article_embedding_80 = article_embedding_80.copy()

    recommandations = []
    for _model in models :
        if "Cb" in _model :

            ###########################################################################################################
            #######    Création  de la liste de Catégories d'articles et de Rating pour chaque utilisateur      #######
            ###########################################################################################################
            _user_rating_categ = _data.groupby(by="user_id").agg(lambda x : list(x)).reset_index()
            _user_rating_categ = _user_rating_categ[["user_id","category_id","Rating"]]

            ###########################################################################################################
            #########    Génération des recommandation d'article et ajout à la liste des recommandations      #########
            ###########################################################################################################
            recommandations.append( content_base_recommandation_par_utilisateur(
                                                                                    user_id , _user_rating_categ , _articles_metadata, n 
                                                                                ).reset_index(drop=True)[["article_id"]] )

        elif "Svd" in _model :
            ###########################################################################
            #########    Calcul du ratings moyen pour chaque utilisateur      #########
            ###########################################################################
            _user_mean = _data.groupby(by="user_id").mean()["Rating"].values

            #####################################################################################
            #########  création de la rating matrix ( rating count par utilisateur )    #########
            #####################################################################################
            _rating_matrix = process_rating_matrix(_data, _user_mean)

            ##########################################################################################################
            #########  class SVD qui servira à la décomposition de la rating_matrix en valeurs simgulière    #########
            ##########################################################################################################
            svd = SVD( _user_mean , _articles_metadata)

            ##########################################################################################
            #########       Décomposition en valeurs simgulières de la rating matrix         #########
            ##########################################################################################
            svd.fit( _rating_matrix )

            ###########################################################################################################
            #########    Génération des recommandation d'article et ajout à la liste des recommandations      #########
            ###########################################################################################################
            recommandations.append( svd.recommend( user_id = user_id, n = n).reset_index(drop=True)[["article_id"]] )

        elif "Distance" in _model :

            ###########################################################################################################
            #########    Génération des recommandation d'article et ajout à la liste des recommandations      #########
            ###########################################################################################################
            recommandations.append( distance_similarité( user_id, _data, _article_embedding_80, _articles_metadata, n = n ).reset_index(drop=True)[["article_id"]] )		
        elif "Cosinus" in _model :

            ###########################################################################################################
            #########    Génération des recommandation d'article et ajout à la liste des recommandations      #########
            ###########################################################################################################
            recommandations.append( cosinus_similarité( user_id, _data, _article_embedding_80, _articles_metadata, n = n ).reset_index(drop=True)[["article_id"]] )	


    return recommandations	



