using System;

// dotnet add package pythonnet
using Python.Runtime;
// python --version
Runtime.PythonDLL = "python311.dll";
PythonEngine.Initialize();

dynamic tf = Py.Import("tensorflow");
dynamic np = Py.Import("numpy");
dynamic model = tf.keras.models.load_model("CheckpointSave/model_94_74.keras");

int index = 0;

// pip install pillow
dynamic list = new PyList();
//ALUNOS
// list.append(tf.keras.utils.load_img("tests/sala/tavares.png"));
// list.append(tf.keras.utils.load_img("tests/sala/vini.png"));
// list.append(tf.keras.utils.load_img("tests/sala/lander.png"));
// list.append(tf.keras.utils.load_img("tests/sala/felipe.png"));
// list.append(tf.keras.utils.load_img("tests/sala/rosa.png"));
// list.append(tf.keras.utils.load_img("tests/sala/mateus.png"));
// list.append(tf.keras.utils.load_img("tests/sala/trevis.png"));
// list.append(tf.keras.utils.load_img("tests/sala/benhur.png"));
// list.append(tf.keras.utils.load_img("tests/sala/eliana.png"));
// list.append(tf.keras.utils.load_img("tests/sala/marcos.png"));
// list.append(tf.keras.utils.load_img("tests/sala/eric.png"));
list.append(tf.keras.utils.load_img("tests/sala/emylli.png"));

//ATORES
// list.append(tf.keras.utils.load_img("tests/atores/will.png"));
// list.append(tf.keras.utils.load_img("tests/atores/tomCruise.png"));

dynamic data = np.array(list);
dynamic result = model.predict(data);

for (int i = 0; i < 17; i++) {
    if(result[i].As<float>() > result[index].As<float>())
        index = i;
}

string[] names = {"Angelina Jolie", "Brad Pitt", "Denzel Washington", "Hugh Jackman", "Jennifer Lawrence", "Jhonny Depp", "Kate Winslet", 
                  "Leonardo DiCaprio", "Megan Fox", "Natalie Portman", "Nicole Kidman", "Robert Downey Jr", "Sandra Bullock", "Scarlett Johansson",
                  "Tom Cruise", "Tom Hanks", "Will Smith"}; 

Console.WriteLine(result[index]);
Console.WriteLine(names[index]);
PythonEngine.Shutdown();
