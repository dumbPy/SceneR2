from SceneR2.core import *
from SceneR2.dataset import MovingObjectData2
from SceneR2.models import CanModel
from SceneR2 import saved_models
from SceneR2.utils import fig2rgb_array
import pkg_resources
from SceneR2.yolov3 import VideoPipeline
from PIL import Image


def process_can_and_video(static_path, csv_name, vid_name, fps=25, **kwargs):
    # Label=0 is a hack as dataset will not find the passed CAN data's label 
    # in out labeled dataset This thing has to be fixed in future iterations
    dataset = MovingObjectData2([os.path.join(static_path, csv_name)], label=0)
    model = CanModel()
    model_path = pkg_resources.resource_filename(saved_models, 'CanModel_31May.pt')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    dataloader = data.DataLoader(dataset)
    
    # Make class prediction
    for x,y in dataloader:
        x=x.float().to(device)
        with torch.no_grad():
            y_prob = model(x)
            y_prob = y_prob.cpu().numpy()
        y_pred = np.argmax(y_prob)

    # Process Image, make plots and save to png file
    # check tight_layout in SceneR2.dataset.BaseObject.plot
    # It is used to eliminate whitespaces and make y_ticks amd labels inside
    all_axes = dataset.plot(0, supressPostABA=False, all_columns=True, 
                            tight_layout=True, verbose=True)
    imgs = [fig2rgb_array(axes.flat[0].get_figure()) for axes in all_axes]
    img = np.concatenate(imgs)
    Image.fromarray(img).save(os.path.join(static_path, 'can_slider.png'))
    
    # Process video, add bounding box
    pip = VideoPipeline()
    vid_file_path = pip.vid2vid(os.path.join(static_path, vid_name),
                    static_path, fps=25, **kwargs)
    return y_pred




