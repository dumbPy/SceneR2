
from ..core import *
from .models import Darknet
from .utils.utils import *
from .utils.datasets import VideoDataset
from torchvision.transforms.transforms import ToTensor
from imageio import get_reader
from torch.utils.data import DataLoader
from matplotlib.ticker import NullLocator
import matplotlib
# matplotlib.use('agg')
import random
from PIL import Image
import pkg_resources

from . import config
from . import weights

class VideoPipeline:
    def __init__(self, img_size=416, batch_sz=8, nms_thres=0.4, conf_thres=0.8, device=None, n_cpu=0):
        """
        Paramaters
        ----------
        img_size:       Size to resize each frame to before inference
        batch_sz:       Batch Size for batch processing
        nms_thres:      IOU Threshold for Non Maximum Supression
        conf_thres:     Confedence Threshold for Non Max Supression
        device:         device to use eg. torch.device('cuda:0')
        """
        self.img_size = img_size
        self.n_cpu = n_cpu

        self.config_path = pkg_resources.resource_filename(
                            config.__name__, 'yolov3.cfg')

        self.weights_path = pkg_resources.resource_filename(
                            weights.__name__, 'yolov3.weights')        

        self.classes_path = pkg_resources.resource_filename(
                            config.__name__, 'coco.names')

        self.classes = load_classes(self.classes_path)
        self.batch_sz = batch_sz
        self.nms_thres,self.conf_thres = nms_thres,conf_thres
        
        # Set up model
        self.model = Darknet(self.config_path, img_size=self.img_size)
        self.model.load_weights(self.weights_path)
        if device is None: device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        
    def vid2vid(self, source_path, dest_folder):
        """
        Paramaters
        -----------
        source_path:    Path of the source video for object detection
        dest_path:      Path of the final video woth bounding boxes
        
        Returns
        ------------
        dest_path
        """

        assert not os.path.isfile(dest_folder), "dest_folder should be a directory not a file"
                
        os.makedirs(dest_folder, exist_ok=True)
        dest_filename = os.path.split(source_path)[1]
        dest_filename = os.path.join(dest_folder, dest_filename)

        classes = self.classes
        img_detections = []
        
        dataloader = DataLoader(
        VideoDataset(source_path, img_size=self.img_size),
        batch_size=self.batch_sz,
        shuffle=False,
        num_workers=self.n_cpu,
        )
        model = self.model
        # Inference
        for imgs in tqdm.tqdm(dataloader, leave=False):
            with torch.no_grad():
                imgs = imgs.float().to(device)
                detections = model(imgs)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            img_detections.extend(detections)


        # Bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        video_writer = imageio.get_writer(dest_filename, fps=25)
        for i, img in tqdm.tqdm(enumerate(get_reader(source_path)), leave=False):
            detections = img_detections[i]
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            
            size = self.img_size

            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            video_writer.append_data(data)
            plt.close()
            # filename = os.path.join(dest_path, 'tmp', str(i).zfill(3))+'.png'
            # plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)
            # plt.close()

        video_writer.close()
        with open(f"{dest_filename[:-3]}_predictions", "wb") as f:
            pickle.dump(img_detections, f)
        return dest_filename