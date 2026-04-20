import torch
import torch.nn as nn
import csv, math

dataset_csv = './tokenized_dataset.csv'
test_train_split = 0.7
num_epochs = 3
learning_rate = 0.06
data_input_size = 30
hidden_size = 15



dataset = []
with open(dataset_csv, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        tokens_list = []
        for token in row[2].split("|"):
            try:
                tokens_list.append(int(token) * 1.0)
            except:
                pass
        dataset.append([tokens_list, torch.tensor(int(row[1]))])

total = len(dataset)
print("total number of items = " + str(total))

temp = [0] * data_input_size

for data_entry in dataset:
    data_entry[0] = torch.tensor((data_entry[0] + temp)[0:data_input_size])

split_index = math.floor(len(dataset) * test_train_split)

split_dataset = {}
split_dataset["train"] = dataset[0:split_index]
split_dataset["test"] = dataset[split_index+1:-1]



train_loader = torch.utils.data.DataLoader(dataset=split_dataset["train"], 
                                           batch_size=16, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=split_dataset["test"], 
                                          batch_size=16, 
                                          shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=data_input_size, hidden_dim=hidden_size, output_dim=2):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        

    def forward(self,input):
        out = self.layer1(input)
        out = self.sigmoid(out)
        out = self.layer2(out)
        return out

model = NeuralNetwork()

loss_calc = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

itterator = 0
for epoch in range(num_epochs):
    for _, (examples, labels) in enumerate(train_loader):
        itterator += 1
        examples = examples.view(-1, data_input_size).requires_grad_()
        optimiser.zero_grad()
        predictions = model.forward(examples)
        loss = loss_calc(predictions, labels)
        loss.backward()
        optimiser.step()
        if itterator % 1 == 0:
            correct = 0
            total = 0
            for testcases, labels in test_loader:
                testcases = testcases.view(-1, data_input_size).requires_grad_()
                outputs = model.forward(testcases)
                _, prediction = torch.max(outputs.data, 1)

                total += labels.size(0)

                # Total correct predictions
                correct += (prediction == labels).sum()

            accuracy = 100 * correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(itterator, loss.item(), accuracy))

