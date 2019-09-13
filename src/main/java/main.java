import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.primitives.Doubles;
import one.util.streamex.IntStreamEx;
import one.util.streamex.StreamEx;
import org.apache.commons.codec.binary.Base64;
import com.google.common.primitives.Bytes;
import com.google.common.primitives.Ints;
import com.google.gson.*;
import kong.unirest.HttpResponse;
import kong.unirest.JacksonObjectMapper;
import kong.unirest.Unirest;
import net.dongliu.requests.Parameter;
import net.dongliu.requests.Requests;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_java;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.datavec.image.loader.NativeImageLoader;
import org.json.JSONException;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;


import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import one.util.streamex.*;

public class main {
    private static byte[] buf;

    static {
        Loader.load(opencv_java.class);

    }


    public  static void main(String[] args) throws Exception {
         String[] TensorCocoClasses=new String[]{
            "background",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "12",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "26",
            "backpack",
            "umbrella",
            "29",
            "30",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "45",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "66",
            "dining table",
            "68",
            "69",
            "toilet",
            "71",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "83",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"};
        Gson gson = new Gson();
        Mat img = org.bytedeco.opencv.global.opencv_imgcodecs.imread("/home/danhyal/motor.jpg");
//        org.bytedeco.opencv.global.opencv_imgproc.resize(img,img,new Size(1200,1200));

        ByteBuffer temp=img.getByteBuffer();
        byte[] arr = new byte[temp.remaining()];
        temp.get(arr);
        opencv_imgcodecs.imencode(".jpg", img, arr);
        String encoded= Base64.encodeBase64String(arr);

        org.json.simple.JSONObject json=new org.json.simple.JSONObject();
        org.json.simple.JSONArray oof=new org.json.simple.JSONArray();
        JSONObject b64=new JSONObject();
        b64.put("b64",encoded);
        oof.add(b64);
        json.put("instances",oof);
        String server_url = "http://localhost:8501/v1/models/nasnet:predict";
        JSONParser jsonParser=new JSONParser();

            final long startTime = System.currentTimeMillis();

            String response = Requests.post(server_url)
                .jsonBody(json).socksTimeout(50000)
                .send().readToText();
        Object obj=jsonParser.parse(response);
        JSONObject jobj=(JSONObject) obj;
        JSONArray content=(JSONArray) jobj.get("predictions");
        Iterator i = content.iterator();

        JSONObject predictions = (JSONObject) i.next();
        int num_detections = ((Double) predictions.get("num_detections")).intValue();
        double[] detection_classes = gson.fromJson(predictions.get("detection_classes").toString(),(Type)double[].class);
        List<Double> detection_scores=new ArrayList<>();
        double[] detection_scoress =gson.fromJson(predictions.get("detection_scores").toString(),(Type)double[].class);
        for (double x:detection_scoress){
            if (x!=0.0){
                detection_scores.add(x);
            }
        }
        Double[][] detection_boxess=gson.fromJson(predictions.get("detection_boxes").toString(), (Type) Double[][].class);
        List<List<Double>> detection_boxes= StreamEx.of(detection_boxess).map(a -> DoubleStreamEx.of(a).boxed().toList()).toList();
        System.out.println(Arrays.toString(detection_classes));
        for (int j=0;j<num_detections;j+=1){
            double confidance=detection_scores.get(j);
            if (confidance>0.7){
                int top= (int) (detection_boxes.get(j).get(0)*img.rows());
                int left=(int)(detection_boxes.get(j).get(1)*img.cols());
                int bottom=(int)(detection_boxes.get(j).get(2)*img.rows());
                int right=(int)(detection_boxes.get(j).get(3)*img.cols());
                org.bytedeco.opencv.global.opencv_imgproc.rectangle(img,new Point(left,top),new Point(right,bottom), Scalar.GREEN);
                org.bytedeco.opencv.global.opencv_imgproc.putText(img, TensorCocoClasses[(int) detection_classes[j]],new Point(left,top),1,1,Scalar.RED);
                System.out.println(String.format("%s,%s %s,%s",left,top,right,bottom));


            }
        }
        final long endTime = System.currentTimeMillis();

        org.bytedeco.opencv.global.opencv_imgcodecs.imwrite("/home/danhyal/motorp.jpg",img);
        System.out.println("Total execution time: " + (endTime - startTime));








    }


}

