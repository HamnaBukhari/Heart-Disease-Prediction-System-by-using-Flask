from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

file=open('model.pkl','rb')
model=pickle.load(file)
file.close()

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method=='POST':
        myDict=request.form
        age=int(myDict['age'])
        anaemia=int(myDict['anaemia'])
        creatinine_phosphokinase=int(myDict['creatinine_phosphokinase'])
        diabetes=int(myDict['diabetes'])
        ejection_fraction=int(myDict['ejection_fraction'])
        high_blood_pressure=int(myDict['high_blood_pressure'])
        platelets=int(myDict['platelets'])
        serum_creatinine=int(myDict['serum_creatinine'])
        serum_sodium=int(myDict['serum_sodium'])
        sex=int(myDict['sex'])
        smoking=int(myDict['smoking'])
        time=int(myDict['time'])
        inputFeatures=[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]
        infProb=model.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html',inf=round(infProb*100))
    
        
    
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)