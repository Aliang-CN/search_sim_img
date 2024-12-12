FROM linxingliang/python-cv:0.0.2

ADD /search_sim_img/ /app/search_sim_img
WORKDIR /app/search_sim_img/

CMD ["python","steaming_main.py"]
