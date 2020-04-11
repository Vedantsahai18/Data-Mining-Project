import numpy as np
from sklearn.svm import SVC

def makeLine(xx, yy):
    leftPoint = {'x': xx[0], 'y': np.round(yy[0], 4)}
    rightPoint = {'x': xx[1], 'y': np.round(yy[1], 4)}
    return leftPoint, rightPoint


def getColors(clf, xx, yy_up, yy_down):
    colors = ['#000000']
    predicted = clf.predict([[xx[0], yy_up[0]]])
    if np.max(predicted) == 1:
        colors.append('#FF0000')
    else:
        colors.append('#0000FF')

    predicted = clf.predict([[xx[0], yy_down[0]]])
    if np.max(predicted) == 1:
        colors.append('#FF0000')
    else:
        colors.append('#0000FF')
    
    return colors


def svm(data, eps=1e-3):
    x = np.array(data["x"], dtype='float')
    y = np.array(data["y"], dtype='float')
    c = float(data['c'])
    D = np.array([x, y]).T
    labels = np.array(data["labels"], dtype=np.int32)
    uniqueLabels = np.unique(labels)
    if len(uniqueLabels) != 2:
        zero = {'x': 0, 'y': 0}
        return {
            "boundaryLine": [zero] * 2,
            "upperLine": [zero] * 2,
            "lowerLine": [zero] * 2,
            "colors": ['#000000', '#000000', '#000000'],
            "accuracy": 'N/A'
        }

    clf = SVC(kernel='linear', C=c)
    clf.fit(D, labels)
    acc = 100 * np.sum(clf.predict(D) == labels) / labels.shape[0]
    acc = np.round(acc, 2)

    w = clf.coef_[0]
    m = -w[0] / (w[1] + eps)
    b = -clf.intercept_[0] / (w[1] + eps)

    xx = np.array([np.floor(np.min(x)) - 1, np.ceil(np.max(x)) + 1])
    yy = m * xx + b

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + m ** 2) * margin
    yy_up = yy + np.sqrt(1 + m ** 2) * margin
    
    leftPoint, rightPoint = makeLine(xx, yy)
    upperLeft, upperRight = makeLine(xx, yy_up)
    downLeft, downRight = makeLine(xx, yy_down)
    colors = getColors(clf, xx, yy_up, yy_down)
    
    output_data = {
        "boundaryLine": [leftPoint, rightPoint],
        "upperLine": [upperLeft, upperRight],
        "lowerLine": [downLeft, downRight],
        "colors": colors,
        "accuracy": f'{acc}%'
    }

    return output_data
