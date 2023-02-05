import torch
import torch.optim as optim

def train(net, dataloader, criterion, optimizer, _cls = False):
    # train the network
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, bbox = data

        # zero the parameter gradients (still not sure what this is really?)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        if _cls:
            loss = criterion(outputs, bbox, labels)

        else:
            loss = criterion(outputs, bbox)
        loss.backward()
        optimizer.step()

        # print stats
        running_loss += loss.item()

    return running_loss


def validation(net, dataloader, criterion, _cls = False):
    running_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, bbox = data

            # forward + loss
            outputs = net(inputs.float())
            if _cls:
                loss = criterion(outputs, bbox, labels)

            else:
                loss = criterion(outputs, bbox)


            # print stats
            running_loss += loss.item()
    return running_loss

def train_and_validate(net, epochs, trainloader, validloader, criterion, optimizer, k = 10, _cls = False):
    train_results = dict()
    valid_results = dict()
    for epoch in range(epochs):
        train_results[epoch] = train(net, trainloader, criterion, optimizer, _cls = _cls)
        valid_results[epoch] = validation(net, validloader, criterion, _cls = _cls)

        if (epoch + 1) % k == 0:
            print(f'EPOCH: {epoch + 1} training loss: {train_results[epoch]:.3f} validation loss: {valid_results[epoch]:.3f}')
    return {'train':train_results, 'valid':valid_results}
