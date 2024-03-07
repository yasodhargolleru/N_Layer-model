from flask import Flask, render_template_string, request, redirect, url_for, jsonify, make_response
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import seaborn as sns
import io
import base64
import logging



app = Flask(__name__)
app.logger.setLevel(logging.INFO)
#Form template
form_template = """
#Form template

<!DOCTYPE html>
<html>
<head>
    <title>N-layer Model</title> <!-- Browser Tab Title -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f8f9fa;
        }
        .container {
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 .125rem .25rem rgba(0, 0, 0, .075);
            padding: 20px;
            margin-top: 20px;
        }
        .title { /* Added class for the new title */
            text-align: center;
            margin-bottom: 20px;
        }
        .section-title {
            margin-bottom: 15px;
        }
        .configuration-section, .image-section {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 .125rem .25rem rgba(0, 0, 0, .075);
            border-radius: 5px;
        }
        .form-group label, .form-check-label {
            color: #495057;
            font-weight: bold;
        }
        .btn-primary {
            background-color: #007bff;
            border-color: #007bff;
        }
        .btn-primary:hover {
            background-color: #0056b3;
            border-color: #0056b3;
        }
        input.form-control, input[type="submit"] {
            border-radius: 0.25rem;
            border: 1px solid #ced4da;
            padding: 0.375rem 0.75rem;
            font-size: 1rem;
            line-height: 1.5;
            color: #495057;
            background-color: #fff;
            background-clip: padding-box;
            transition: border-color .15s ease-in-out,box-shadow .15s ease-in-out;
        }
        input.form-control:focus, input[type="submit"]:focus {
            border-color: #80bdff;
            outline: 0;
            box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .description {
            margin-top: 20px;
            text-align: center;
        }
    </style>
    <script>
        function updateDescription() {
            var probModel = document.getElementById('prob');
            var description = document.getElementById('model-description');
            
            if (probModel.checked) {
                description.innerHTML = 'Probabilistic model is to model Non-Strategic Attackers.';
            } else {
                description.innerHTML = 'Strategic model is to model Strategic Attackers.';
            }
        }
    </script>
</head>
<body>
    <div class="container shadow-sm bg-white rounded">
        <h2 class="title">N-layer Model </h2> 
        <div class="row">
            <div class="col-md-6">
                <h3 class="section-title">Configuration</h3>
                <div class="configuration-section">
                    <form action="/" method="post" class="rounded">
                        <div class="form-group">
                            <label>Model Type:</label><br>
                            <div class="form-check">
                                <input class="form-check-input" type="radio" name="model_type" id="prob" value="prob" checked onchange="updateDescription()">
                                <label class="form-check-label" for="prob">Probabilistic Model</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="radio" name="model_type" id="stra" value="stra" onchange="updateDescription()">
                                <label class="form-check-label" for="stra">Strategic Model</label>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="total_layers">Total Layers:</label>
                            <input type="number" class="form-control" name="total_layers" id="total_layers" required>
                        </div>
                        <div class="form-group">
                            <label for="C_bar_init">C:</label>
                            <input type="number" class="form-control" name="C_bar_init" id="C_bar_init" required>
                        </div>
                        <input type="submit" value="Submit" class="btn btn-primary">
                    </form>
                </div>
            </div>
            <div class="col-md-6">
                <h3 class="section-title">Game Tree</h3>
                <div class="image-section">
                    <img src="{{ url_for('static', filename='image.png') }}" alt="Model Configuration Image" class="img-fluid rounded">
                    <div id="model-description" class="description">Probabilistic model is to model non-strategic attackers.</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# Results_template with Bootstrap elements
results_template = """
<!DOCTYPE html>
<html>
<head>
    <title>N-layer App Results</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background-color: #FFFFFF;
        }
        .bg-light {
            background-color: #FFFFFF !important;
        }
        .container {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            padding: 30px;
        }
        form {
            background-color: rgba(255, 245, 225, 0.95);
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            border-radius: 5px;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        }
        #total_layers, #C_bar_init {
            max-width: 300px;
        }
        .image-container {
            max-width: 100%;
            margin-top: 50px;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
        }
        .model-description {
            margin-top: 20px;
            text-align: center;
            font-size: 16px;
            color: #495057;
        }
    </style>
    <script>
        function updateDescription() {
            var probModel = document.getElementById('prob');
            var description = document.getElementById('model-description');
            
            if (probModel.checked) {
                description.innerHTML = 'Probabilistic model is to model Non-Strategic Attackers.';
            } else {
                description.innerHTML = 'Strategic model is to model Strategic Attackers.';
            }
        }
    </script>
</head>

<body class="bg-light">
    <div class="container mt-5">
        <h2 class="text-center">N-layer Model Results</h2>

        <div class="row mt-4">
            <div class="col-md-6">
                <div class="bg-white p-4 rounded shadow-sm">
                    <h3>Configuration</h3>
                    <form action="/" method="post">
                        <div class="form-group">
                            <label for="model_type">Model Type:</label><br>
                            <div class="form-check">
                                <input class="form-check-input" type="radio" name="model_type" id="prob" value="prob" {% if model_type == 'prob' %}checked{% endif %} onchange="updateDescription()">
                                <label class="form-check-label" for="prob">Probabilistic Model</label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="radio" name="model_type" id="stra" value="stra" {% if model_type == 'stra' %}checked{% endif %} onchange="updateDescription()">
                                <label class="form-check-label" for="stra">Strategic Model</label>
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="total_layers">Total Layers:</label>
                            <input type="number" class="form-control" name="total_layers" id="total_layers" value="{{ total_layers }}" required>
                        </div>
                        <div class="form-group">
                            <label for="C_bar_init">C:</label>
                            <input type="number" class="form-control" name="C_bar_init" id="C_bar_init" value="{{ C_bar_init }}" required>
                        </div>
                        <input type="submit" value="Submit" class="btn btn-primary">
                    </form>
                </div>
            </div>

            <div class="col-md-6">
                <div class="bg-white p-4 rounded shadow-sm">
                    <h3>Game Tree</h3>
                    <div class="image-container">
                        <img src="{{ url_for('static', filename='image.png') }}" alt="Model Configuration Image" class="img-fluid rounded">
                    </div>
                    <div id="model-description" class="model-description">
                        Probabilistic model is to model Non-Strategic Attackers. <!-- Default text -->
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="bg-white p-4 rounded shadow-sm">
                    <h3>Optimization Outputs</h3>
                    <p><strong>Model:</strong> {{ model }}</p>
                    <p><strong>Solutions:</strong> {{ solutions }}</p>
                    <h4>Values:</h4>
                    <ul>
                        <li><strong>s:</strong> {{ s }}</li>
                        <li><strong>beta:</strong> {{ beta }}</li>
                        <li><strong>alpha:</strong> {{ alpha }}</li>
                        <li><strong>theta:</strong> {{ theta }}</li>
                        <li><strong>gamma:</strong> {{ gamma }}</li>
                        <li><strong>cost:</strong> {{ cost }}</li>
                        <li><strong>C_bar:</strong> {{ C_bar }}</li>
                    </ul>
                </div>
            </div>
            
            <div class="col-md-6 plot-container">
                <div class="bg-white p-4 rounded shadow-sm">
                    <h3>Optimal investment (%) across layers</h3>
                    <img src="data:image/png;base64,{{ plot_url }}" alt="N-layer Results Plot" class="img-fluid rounded">

                </div>
            </div>
        </div>
    </div>
</body>
</html>

"""

# Initialization functions
class instance_nLY():
    
    def __init__(self, s=[], beta=[], alpha=[], theta=[], gamma=[], cost=[], C_bar=[]):
        
#         self.delta = delta
        self.s = s
        self.beta = beta
        self.alpha = alpha
        
        self.theta = theta
        self.gamma = gamma 
        self.cost = cost
        self.C_bar = C_bar


    def print_values(self):
        
        print("s = ", self.s)
        
        print("beta = ", self.beta)
        
        print("alpha = ", self.alpha)
       
        print("theta = ", self.theta)
        
        print("gamma = ", self.gamma)
        
        print("cost = ", self.cost)

        print("C_bar = ", self.C_bar)

def flatten_list(_2d_list):
    flat_list = []
    
# Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
# If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def objective_prob(Y,_nLayers):
    global obj2
    
    f = [None] * _nLayers
    
    for i in range(len(f)):
        f[i] = np.exp(-1 * sum([obj2.gamma[i-k]*obj2.theta[k]*Y[k] for k in range(i+1)])) 
            
    
    raw_risk = [_s*_beta*_alpha for _s, _beta, _alpha in zip(obj2.s, obj2.beta, obj2.alpha)]
    
    return sum(a * b for a, b in zip(raw_risk, f)) 

def objective_stra(Y,_nLayers):
    global obj2
    f = [None] * _nLayers
    
    for i in range(len(f)):
        f[i] = np.exp(-1 * sum([obj2.gamma[i-k]*obj2.theta[k]*Y[k] for k in range(i+1)])) 
            
    raw_risk = [_s*_alpha for _s, _alpha in zip(obj2.s, obj2.alpha)]
    return max(a * b for a, b in zip(raw_risk, f)) 

def constraint(Y, obj2):
    return obj2.C_bar - sum(a * b for a, b in zip(obj2.cost, Y))


def get_numerical_sol(init_val, _model, _nLayers,obj2):
    Y0 = init_val
    
    b = (0.0,None) # bound
    bnds = (b,)*_nLayers
#     bnds = (b, b, b, b, b)

    con1 = {'type': 'eq', 'fun': lambda Y: constraint(Y, obj2)}
    cons = ([con1])
    
    if _model == 'prob':
        solution= minimize(lambda Y: objective_prob(Y, _nLayers), Y0, method='SLSQP', bounds=bnds, constraints=cons)

        x = solution.x
        x = [round(i,4) for i in x] 
        obj_value = round(objective_prob(x,_nLayers),4)
        
    elif _model == 'stra':
        solution = minimize(lambda Y: objective_stra(Y, _nLayers), Y0, method='SLSQP', bounds=bnds, constraints=cons)
                         
        x = solution.x
        x = [round(i,4) for i in x] 
        obj_value = round(objective_stra(x,_nLayers),4)
    else:
        print("Something is wrong here......(get_numerical_sol)")
    
    return flatten_list([obj_value, x])

def addRow(df,ls):
    """
    Given a dataframe and a list, append the list as a new row to the dataframe.

    :param df: <DataFrame> The original dataframe
    :param ls: <list> The new row to be added
    :return: <DataFrame> The dataframe with the newly appended row
    """
    numEl = len(ls)

    newRow = pd.DataFrame(np.array(ls).reshape(1,numEl), columns = list(df.columns))
    df = pd.concat([df, newRow], ignore_index=True)


    return df

def get_full_sol(_nLayers, whichmodel, obj2,vars_col):
    intial_sol = [3]*_nLayers
    solutions = get_numerical_sol(intial_sol, whichmodel, _nLayers, obj2)
    
# Adjust the length of solutions to match the number of columns in solution_df
    required_length = len(vars_col)  # Assuming vars_col is globally accessible
    while len(solutions) < required_length:
        solutions.append(0)
    
    return solutions[:required_length]

def initialization(_nLayers,C_bar_init):
    gam = 0.5 

    s_init = [500 for i in range(1, _nLayers+1)]
    alpha_init = [0.5]*_nLayers
    beta_init = [1/_nLayers]*_nLayers
    theta_init = [0.4]*_nLayers 
    cost_init = [1 for i in range(1, _nLayers+1)]
    gamma_init = [1]+[gam**i for i in range(1, _nLayers)]
    
    obj_base = instance_nLY(s = s_init, alpha = alpha_init, beta = beta_init, theta = theta_init, 
                        cost = cost_init, gamma = gamma_init, C_bar = C_bar_init)


    return obj_base



def generate_plot(whichmodel, solution_df, total_layers, obj2):
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url


# APP
@app.route('/', methods=['GET', 'POST'])
def index():
    global obj2
    

    if request.method == 'POST':
        model_type = request.form.get('model_type')
        total_layers = int(request.form.get('total_layers'))
        C_bar_init = float(request.form.get('C_bar_init'))
        

        app.logger.info(f"Total Layers: {total_layers}")
        app.logger.info(f"C_bar_init: {C_bar_init}")
        # Use the provided logic to compute results and generate the plot
        vars_col = ['obj_value'] + ['y' + str(i+1) for i in range(total_layers)]
        model_set = ['prob', 'stra']
        prob_solutions = None
        stra_solutions = None

        labels = ['Layer ' + str(i+1) for i in range(total_layers)]
        plot_urls = {} 

        for m in range(len(model_set)):
            whichmodel = model_set[m]
            solution_df = pd.DataFrame(columns=vars_col)

            for i in range(total_layers):
                _nLayers = i + 1
                obj_base = initialization(_nLayers,C_bar_init)
                obj2 = copy.deepcopy(obj_base)
                solutions = get_full_sol(_nLayers, whichmodel, obj2,vars_col)
                solution_df = addRow(solution_df, solutions)
            
            solution_df["Layers"] = [i for i in range(1, total_layers+1)] 
    
            if whichmodel == 'prob':
                prob_solutions = solutions
            elif whichmodel == 'stra':
                stra_solutions = solutions
            else:
                print("Something is wrong.")

# Plotting code
            finaldata = pd.DataFrame()
            
            for i in range(0, total_layers):
                new_name = 'invest'+str(i+1)
                old_name = 'y'+str(i+1)
                finaldata[new_name] = obj2.cost[i]*solution_df[old_name]
                
            data_perc = 100*finaldata[finaldata.columns.values.tolist()].divide(finaldata[finaldata.columns.values.tolist()].sum(axis=1), axis=0)
            data_perc["Layers"] = [i for i in range(1, total_layers+1)]
            app.logger.info(f"Type of data_perc['Layers']: {type(data_perc['Layers'])}")
            if hasattr(data_perc['Layers'], 'dtype'):
                app.logger.info(f"dtype of data_perc['Layers']: {data_perc['Layers'].dtype}")

            app.logger.info(f"NaN values in 'Layers': {data_perc['Layers'].isna().any()}")
        

            for i in range(1, total_layers + 1):
                col = 'invest' + str(i)
                data_perc[col] = pd.to_numeric(data_perc[col], errors='coerce')
                app.logger.info(f"Data type of {col}: {data_perc[col].dtype}")
                app.logger.info(f"NaN values in '{col}': {data_perc[col].isna().any()}")
                app.logger.info(f"Infinite values in '{col}': {np.isinf(data_perc[col]).any()}")


            


            plt.stackplot(data_perc['Layers'],
                  [data_perc['invest'+str(i+1)] for i in range(total_layers)],
                  labels=labels, 
                  colors=sns.color_palette(("Greys"), total_layers),
                  alpha=1, edgecolor='grey')
            
            plt.margins(x=0)
            plt.margins(y=0)
            plt.rcParams['font.size'] = 15
            if whichmodel == 'prob':
                plt.ylabel("Budget allocation (%)", fontsize=18)
                plt.xlabel("Number of layers for LRA-PR Model", fontsize=18)
            if whichmodel == 'stra':
                plt.xlabel("Number of layers for LRA-SR Model", fontsize=18)
            
            plt.legend(fontsize=14, markerscale=2, loc='center left', bbox_to_anchor=(1.2, 0.5),
                       fancybox=True, shadow=True, ncol=1)

# Convert the plot to a PNG image in base64 format
            plot_urls[whichmodel] = generate_plot(whichmodel, solution_df, total_layers, obj2)
            
        if model_type == 'prob':
            display_solutions = prob_solutions
            display_model = 'prob'
        else:
            display_solutions = stra_solutions
            display_model = 'stra'

        return render_template_string(
            results_template,
            model_type=model_type,
            plot_url=plot_urls[display_model],
            total_layers=total_layers,
            C_bar_init=C_bar_init,
            model=display_model,
            solutions=str(display_solutions),
            s=str(obj2.s),
            beta=str(obj2.beta),
            alpha=str(obj2.alpha),
            theta=str(obj2.theta),
            gamma=str(obj2.gamma),
            cost=str(obj2.cost),
            C_bar=str(obj2.C_bar))

    return render_template_string(form_template)




if __name__ == "__main__":
    app.run(debug=True, port=5000)