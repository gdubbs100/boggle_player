import matplotlib.pyplot as plt
import matplotlib.patches as pac

def createBBoxPlot(bboxes, color = 'red'):
    """Sets out a series of bounding boxes and adds to an existing plot"""
    for bb in bboxes:
        rectangle = pac.Rectangle(
            (bb[0], bb[1]),
            abs(bb[2] - bb[0]),
            abs(bb[3] - bb[1]),
            fc = 'none',
            ec=color)
        plt.gca().add_patch(rectangle)

def bboxPlot(img, true_bb, pred_bb = False, prediction = False):
    """
    Creates an image plot with a base image and associated bounding boxes. If modelling, create predicted bboxes as well.
    """

    # plot the image
    plt.axes()
    plt.imshow(
        img.detach().numpy(),
        cmap = 'Greys'
    )

    createBBoxPlot(true_bb)

    if prediction:
        createBBoxPlot(pred_bb, 'green')
    
    plt.show()

def plotODResults(img, bbox, net = False, prediction = False):
    """
    takes image and bbox, network and prediction optional - but both must be present to run prediction.
    net is False, or a pytorch network
    """

    if prediction:
        pred_bbox, pred_labels = net(img.float())
        bboxPlot(
            img = img.squeeze().permute(1, 2, 0), 
            true_bb = bbox, 
            pred_bb = pred_bbox.detach().numpy().reshape(16, 4),
            prediction = prediction
            )
    else:
        bboxPlot(img, bbox)

def plotBB(image_key, all_data, format = 'coco'):

    """Plots image with bbox from raw data dictionary imported from getData"""

    plt.axes()
    plt.imshow(all_data[image_key]['img'].reshape(416, 416, 3))
    if format == 'pascal_voc':
        for bb in all_data[image_key]['bbox']:
                rectangle = pac.Rectangle(
                    (bb[1][0], bb[1][1]),
                    abs(bb[1][2] - bb[1][0]),
                    abs(bb[1][3] - bb[1][1]),
                    fc = 'none',
                    ec="red")
                plt.gca().add_patch(rectangle)
    elif format == 'coco':
        for bb in all_data[image_key]['bbox']:
                rectangle = pac.Rectangle(
                    (bb[1][0], bb[1][1]),
                    bb[1][2],
                    bb[1][3],
                    fc = 'none',
                    ec="red")
                plt.gca().add_patch(rectangle)
    plt.show()   