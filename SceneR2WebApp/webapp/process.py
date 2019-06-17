import matplotlib; matplotlib.use('agg')
from SceneR2.core import *
from SceneR2.dataset import MovingObjectData2, SingleCAN
from SceneR2.models import CanModel
from SceneR2 import saved_models
from SceneR2.utils.dataset import fig2rgb_array
import pkg_resources
from SceneR2.yolov3 import VideoPipeline
from PIL import Image
from SceneR2.errors import NoMovingRelevantObjectInData
from SceneR2.utils.dataset import read_csv_auto
from SceneR2.utils.utils import CanOverlayer

result = {
    0: "Moving Object in front turning left",
    1: "Moving Object in front turning right",
    2: "Moving Object in front braking or stoping"
}

def process_can_and_video(out_folder, can_path, vid_path, fps=25, **kwargs):
    # Label=0 is a hack as dataset will not find the passed CAN data's label 
    # in out labeled dataset This thing has to be fixed in future iterations
    os.makedirs(out_folder, exist_ok=True)
    try: 
        dataset = MovingObjectData2([can_path], label=0)
        model = CanModel()
        model_path = pkg_resources.resource_filename('SceneR2', 'saved_models')
        model_path = os.path.join(model_path, 'CanModel_4Jun.pt')
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
            message = result[y_pred]
    
    # 
    except NoMovingRelevantObjectInData as  e:
        print(str(e))
        message = "Video and CAN uploaded have no moving object as relevant object for ABA reaction (check ABA_typ_SelObj). Our model is trained on moving object based reaction (as examples of other type like stationary object and pedestrian were very few i.e., 2 to 3)"
        # force = True added to CANData so that the init doesn't run when 
        # NoMovingRelevantObjectInData is raised, but we can still plot with force option
        dataset = MovingObjectData2([can_path], label=0, force=True)
        

    # Process Image, make plots and save to png file
    # check tight_layout in SceneR2.dataset.BaseGroup.plot
    # It is used to eliminate whitespaces and make y_ticks amd labels inside
    def plot(   filename, supressPostABA=False, all_columns=True,
                tight_layout=True, verbose=False, **kwargs):
        all_axes = dataset.plot(0, supressPostABA=supressPostABA,
                    all_columns=all_columns, tight_layout=tight_layout, verbose=verbose)
        imgs = [fig2rgb_array(axes.flat[0].get_figure()) for axes in all_axes]
        img = np.concatenate(imgs)
        Image.fromarray(img).save(os.path.join(out_folder, filename))

    # tight layout for slider background
    plot('can_slider.png', verbose=True)
    # Without tight layout for dispaying
    plot('can_image.png', tight_layout=False)
    # Show only columns that are being passed to model
    plot('can_few_cols.png', tight_layout=True, all_columns=False)

    
    # Process video, add bounding box
    pip = VideoPipeline()
    single_can_data = SingleCAN.fromCSV(can_path, supressPostABA=False)
    postprocessor = CanOverlayer(single_can_data)
    vid_dest_path = pip.vid2vid(vid_path,
                    out_folder, fps=25, postprocessor=postprocessor, **kwargs)
    
    len_can = read_csv_auto(can_path).shape[0] # num of observations in CAN file
    slider_height = Image.open(os.path.join(out_folder, 'can_slider.png')).size[1]*0.9
    slider_height = f'{slider_height}px'
    len_video = len(imageio.get_reader(vid_dest_path))
    params = {'len_can':len_can,
              'slider_height':slider_height,
              'len_video':len_video,
              'dy_col':[round(i,3) for i in SingleCAN.fromCSV(can_path).dy]}

    return message, params




