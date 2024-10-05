---
jupyter:
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="jKfA2L5FVjte"}
# **Практика 1: Установка окружения и настройка фреймворков для анализа защищенности ИИ**

По предмету: **Анализ защищенности систем искусственного интеллекта**

Выполнил студент **2 курса** группы **ББМО-02-23**

**Панков Евгений Ромуальдович**
:::

::: {.cell .markdown id="Jnro4R2NVbpn"}
# Цель задания:

Научиться настраивать окружение для работы с анализом защищенности
систем ИИ, установить и проверить работу необходимых библиотек, а также
загрузить и протестировать простую модель для дальнейшего использования
в заданиях.
:::

::: {.cell .markdown id="kvjugxr3WJyd"}
# Задачи:

1.  Установить и настроить окружение на базе Jupyter Notebook.
2.  Установить необходимые библиотеки: TensorFlow, PyTorch, Foolbox,
    CleverHans и другие.
3.  Проверить корректную работу установленных библиотек с помощью
    базового примера.
4.  Загрузить и обучить простую модель нейронной сети для последующего
    использования в дальнейших практиках.
:::

::: {.cell .markdown id="ctOjwhjiWbhk"}
# Шаги выполнения:
:::

::: {.cell .markdown id="-04508wKWh2R"}
## Шаг 1: Установка необходимых библиотек
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="V5KiW5DWWph7" outputId="1fd067d9-c600-412d-f585-fb7a10161fd8"}
``` python
!pip install tensorflow
!pip install torch torchvision
!pip install foolbox
!pip install cleverhans
```

::: {.output .stream .stdout}
    Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.17.0)
    Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)
    Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.3.25)
    Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.6.0)
    Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: h5py>=3.10.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.11.0)
    Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (18.1.1)
    Requirement already satisfied: ml-dtypes<0.5.0,>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.4.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.4.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.1)
    Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.32.3)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (71.0.4)
    Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.4.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.12.2)
    Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.64.1)
    Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.17.0)
    Requirement already satisfied: keras>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.4.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.37.1)
    Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.26.4)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)
    Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (13.8.1)
    Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.0.8)
    Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.12.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2024.8.30)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.7)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.0.4)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.18,>=2.17->tensorflow) (2.1.5)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (2.18.0)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.2.0->tensorflow) (0.1.2)
    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.4.1+cu121)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.19.1+cu121)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch) (3.16.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch) (4.12.2)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.13.3)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.3)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2024.6.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision) (1.26.4)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision) (10.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.5)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)
    Collecting foolbox
      Downloading foolbox-3.3.4-py3-none-any.whl.metadata (7.3 kB)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from foolbox) (1.26.4)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from foolbox) (1.13.1)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from foolbox) (71.0.4)
    Collecting eagerpy>=0.30.0 (from foolbox)
      Downloading eagerpy-0.30.0-py3-none-any.whl.metadata (5.5 kB)
    Collecting GitPython>=3.0.7 (from foolbox)
      Downloading GitPython-3.1.43-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: typing-extensions>=3.7.4.1 in /usr/local/lib/python3.10/dist-packages (from foolbox) (4.12.2)
    Requirement already satisfied: requests>=2.24.0 in /usr/local/lib/python3.10/dist-packages (from foolbox) (2.32.3)
    Collecting gitdb<5,>=4.0.1 (from GitPython>=3.0.7->foolbox)
      Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.24.0->foolbox) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.24.0->foolbox) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.24.0->foolbox) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.24.0->foolbox) (2024.8.30)
    Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->GitPython>=3.0.7->foolbox)
      Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)
    Downloading foolbox-3.3.4-py3-none-any.whl (1.7 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 34.1 MB/s eta 0:00:00
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.3/207.3 kB 9.0 MB/s eta 0:00:00
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.7/62.7 kB 3.5 MB/s eta 0:00:00
    map-5.0.1-py3-none-any.whl (24 kB)
    Installing collected packages: smmap, eagerpy, gitdb, GitPython, foolbox
    Successfully installed GitPython-3.1.43 eagerpy-0.30.0 foolbox-3.3.4 gitdb-4.0.11 smmap-5.0.1
    Collecting cleverhans
      Downloading cleverhans-4.0.0-py3-none-any.whl.metadata (846 bytes)
    Collecting nose (from cleverhans)
      Downloading nose-1.3.7-py3-none-any.whl.metadata (1.7 kB)
    Collecting pycodestyle (from cleverhans)
      Downloading pycodestyle-2.12.1-py2.py3-none-any.whl.metadata (4.5 kB)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from cleverhans) (1.13.1)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from cleverhans) (3.7.1)
    Collecting mnist (from cleverhans)
      Downloading mnist-0.2.2-py2.py3-none-any.whl.metadata (1.6 kB)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from cleverhans) (1.26.4)
    Requirement already satisfied: tensorflow-probability in /usr/local/lib/python3.10/dist-packages (from cleverhans) (0.24.0)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from cleverhans) (1.4.2)
    Requirement already satisfied: easydict in /usr/local/lib/python3.10/dist-packages (from cleverhans) (1.13)
    Requirement already satisfied: absl-py in /usr/local/lib/python3.10/dist-packages (from cleverhans) (1.4.0)
    Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from cleverhans) (1.16.0)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->cleverhans) (1.3.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->cleverhans) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->cleverhans) (4.54.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->cleverhans) (1.4.7)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->cleverhans) (24.1)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->cleverhans) (10.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->cleverhans) (3.1.4)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib->cleverhans) (2.8.2)
    Requirement already satisfied: decorator in /usr/local/lib/python3.10/dist-packages (from tensorflow-probability->cleverhans) (4.4.2)
    Requirement already satisfied: cloudpickle>=1.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow-probability->cleverhans) (2.2.1)
    Requirement already satisfied: gast>=0.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow-probability->cleverhans) (0.6.0)
    Requirement already satisfied: dm-tree in /usr/local/lib/python3.10/dist-packages (from tensorflow-probability->cleverhans) (0.1.8)
    Downloading cleverhans-4.0.0-py3-none-any.whl (92 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 92.3/92.3 kB 4.0 MB/s eta 0:00:00
    nist-0.2.2-py2.py3-none-any.whl (3.5 kB)
    Downloading nose-1.3.7-py3-none-any.whl (154 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 154.7/154.7 kB 8.1 MB/s eta 0:00:00
    nist, cleverhans
    Successfully installed cleverhans-4.0.0 mnist-0.2.2 nose-1.3.7 pycodestyle-2.12.1
:::
:::

::: {.cell .markdown id="sbtjO8pBW2au"}
## Шаг 2: Проверка установки библиотек
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="J_xDX1QlW_SB" outputId="d89c93f1-45ef-4211-d57d-b4362060701d"}
``` python
import tensorflow as tf
import torch
import foolbox
import cleverhans
# Проверка версии TensorFlow
print(f"TensorFlow version: {tf.__version__}")
# Проверка версии PyTorch
print(f"PyTorch version: {torch.__version__}")
# Проверка версии Foolbox
print(f"Foolbox version: {foolbox.__version__}")
# Проверка версии CleverHans
print(f"CleverHans version: {cleverhans.__version__}")
```

::: {.output .stream .stdout}
    TensorFlow version: 2.17.0
    PyTorch version: 2.4.1+cu121
    Foolbox version: 3.3.4
    CleverHans version: 4.0.0-bc3d8048cc7c9cfae800a9be7e4700e9
:::
:::

::: {.cell .markdown id="PDH5Iqp1XFra"}
## Шаг 3: Загрузка и обучение простой модели
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="lyzWnKyAXKZE" outputId="23afc0d9-0ca3-4286-fb10-2264e5b63315"}
``` python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
# Загрузка датасета MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Нормализация данных
train_images = train_images / 255.0
test_images = test_images / 255.0
# Преобразование меток в one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# Создание модели
model = Sequential([
 Flatten(input_shape=(28, 28)),
 Dense(128, activation='relu'),
 Dense(10, activation='softmax')
])
# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=
['accuracy'])
# Обучение модели
model.fit(train_images, train_labels, epochs=5)
# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
```

::: {.output .stream .stdout}
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(**kwargs)
:::

::: {.output .stream .stdout}
    Epoch 1/5
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 16s 7ms/step - accuracy: 0.8753 - loss: 0.4441
    Epoch 2/5
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 17s 5ms/step - accuracy: 0.9643 - loss: 0.1231
    Epoch 3/5
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 5ms/step - accuracy: 0.9758 - loss: 0.0786
    Epoch 4/5
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 11s 6ms/step - accuracy: 0.9823 - loss: 0.0578
    Epoch 5/5
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 5ms/step - accuracy: 0.9861 - loss: 0.0432
    313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9717 - loss: 0.0970
    Test accuracy: 0.9750999808311462
:::
:::

::: {.cell .markdown id="0Zn4sDLUX0P4"}
## Шаг 4: Сохранение модели
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="5Y7HPfa7X4fx" outputId="bb7119ad-e6eb-4d0b-b146-c0da339191a9"}
``` python
model.save('mnist_model.h5')
```

::: {.output .stream .stderr}
    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
:::
:::
