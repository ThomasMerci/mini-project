# import findspark
# findspark.init()
# import json
# from flask import Flask, request, jsonify, render_template,Response
# import os
# import time
# #from metrics import predictions_counter, prediction_duration_histogram, prediction_latency_gauge
# #from prometheus import init_prometheus
# from prometheus_client import Counter
# from prometheus_client import make_wsgi_app
# from werkzeug.middleware.dispatcher import DispatcherMiddleware
# from train_gradient import train_model

# app = Flask(__name__, template_folder=os.path.abspath('templates'))
# app.config['DEBUG'] = True


# @app.route('/train')
# def predict():
#     train_model()
