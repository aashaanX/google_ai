import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

test_idx =  [0,5,100]

print iris.feature_names
print iris.target_names
# print iris.data

# trining data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)
# print train_data

# print "============="
# print train_data

# test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

# vizulaisation Code
# Note : need to install graphviz
# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
#
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          feature_names=iris.feature_names,
#                          class_names=iris.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = graphviz.Source(dot_data)
# graph