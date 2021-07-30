import numpy as np
import pandas as pd
from flask import Flask,render_template,request
import pickle
from sklearn.preprocessing import MinMaxScaler


item_cosine=pickle.load(open('item_cosine.pkl','rb'))
item_matrix=pickle.load(open('item_matrix.pkl','rb'))
user_vecs=pickle.load(open('user_vecs.pkl','rb'))
item_vecs=pickle.load(open('item_vecs.pkl','rb'))
cust_dict=pickle.load(open('cust_dict.pkl','rb'))
itemindex_dict=pickle.load(open('item_dict.pkl','rb'))

item_dict={value:key for key,value in itemindex_dict.items()}


app=Flask(__name__)

@app.route('/')
def homepage():
    return render_template('indextemp.html')


@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        itemname=request.form['item']
        for key,value in itemindex_dict.items():
            if itemname not in itemindex_dict.keys():
                return'{} is not avaiable in this Grocery System.Please Enter the details again.'.format(itemname)
            else:
                top_number=int(request.form['topn'])
                similar_item=[]
                item_id=itemindex_dict[itemname]
                cosscore=list(enumerate(item_cosine[item_id]))
                cosscore=sorted(cosscore , key=lambda x:x[1] , reverse=True)
                cosscore_new=cosscore[1:top_number+2]
                item_idx=[i[0] for i in cosscore_new]
                item_score=[i[1] for i in cosscore_new]
                for i in item_idx:
                    similar_item.append([key for (key,value) in itemindex_dict.items() if value == i])
        return render_template('itemp.html' , item_list=itemindex_dict,item=similar_item)

@app.route('/Submit',methods=['POST','GET'])
def Submit():
    if request.method=='POST':
        purchased_items=[]
        cust_id=request.form['custid']
        for key,value in cust_dict.items():
            if cust_id not in cust_dict.keys():
                return'{} customer id is not avaiable in this System.Please Enter the details again.'.format(cust_id)
        else:
            num_items=int(request.form['numitems'])
            cust_index=[value for (key,value) in cust_dict.items() if key==cust_id][0]
            purchased_index=item_matrix[cust_index,:].nonzero()[1]
            for i in purchased_index:
                purchased_items.append(item_dict[i])
            pref_vec = item_matrix[cust_index,:].toarray()
            pref_vec = pref_vec.reshape(-1) + 1
            pref_vec[pref_vec > 1] = 0
            rec_vector = user_vecs[cust_index,:].dot(item_vecs.T)
            min_max = MinMaxScaler()
            rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] 
            recommend_vector = pref_vec*rec_vector_scaled 
            product_idx = np.argsort(recommend_vector)[::-1][:num_items] 
            rec_list = [] 
            for i in product_idx:
                rec_list.append(item_dict[i])
        return render_template('utemp.html',purchased=purchased_items , rec=rec_list)



if __name__ =='__main__':
    app.run()
