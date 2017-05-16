import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;
import java.util.Random;

import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class Client {

	private static StdDataset datasets;
	private static String trainFile;
	private static String testFile;
	private static String email;
	private static String networkEnds;
	private static String networkRels;
	private static String fileName;

	//configuration
	private static final int TESTNUM = 5;

	public static void main(String[] args) {

		// Default value
		networkEnds = "/home/cs682/TestData/string.ents";
		networkRels = "/home/cs682/TestData/string.rels";

		trainFile = "";
		testFile = "";
		email = "";
		
		// Read command line
		for(int i=0;i<args.length;i++){
			if(args[i].equals("-train")){
				trainFile = args[++i];
			}
			if(args[i].equals("-test")){
				testFile = args[++i];
			}
			if(args[i].equals("-network")){
				networkEnds = args[++i];
				networkRels = args[++i];
			}
			if(args[i].equals("-email")){
				email = args[++i];
			}
		}
		
		if(!trainFile.equals("")){
			

			fileName =  trainFile.substring(trainFile.lastIndexOf("/")+1);
			if(!testFile.equals("")){
				fileName += "_"+testFile.substring(testFile.lastIndexOf("/")+1);
			}
			
			datasets = new StdDataset(networkEnds,networkRels);
			if(testFile.equals("")){
				doCrossValidate();
			}else{
				doTrainTest();
			}
			
			if(email.equals("")){
				System.out.println("Warning: email address empty.");
			}else{
				sendEmail(email);
			}
		}else{
			System.out.println("No Train file!");
		}
	}
	
	private static void sendEmail(String to){
		
	    String host = "smtp.gmail.com";
	    String username = "clinicaltrialpredictor";
	    String password = "classProject";
	    Properties props = new Properties(); 
		props.put("mail.smtp.auth", "true");
		props.put("mail.smtp.starttls.enable", "true");
		props.put("mail.smtp.host", host);   
		props.put("mail.smtp.port", 587);   
	    Session session = Session.getInstance(props);
	    session.setDebug(true);
	    try {
		    MimeMessage msg = new MimeMessage(session);    
		    msg.addRecipient(Message.RecipientType.TO,new InternetAddress(to));    
		    msg.setSubject("Your task is complete.");    
		    msg.setText("Your task " + fileName + " has been finished. Please go to our website to check the results.");    
		    
			Transport.send(msg, username, password);
		} catch (MessagingException e) {
			e.printStackTrace();
		}
	}
	
	private static void doCrossValidate(){
		
		// Read input data
		datasets.readData(trainFile);

		datasets.standardizeData();

		// Prepare Network
		CNNInstance[] ins = new CNNInstance[TESTNUM];
		Thread[] trd = new Thread[TESTNUM];

		//used data
		ArrayList<ArrayList<Double>> d1 = new ArrayList<ArrayList<Double>>(datasets.getDatasets().get(0));
		ArrayList<Integer> gt1 = new ArrayList<Integer>(datasets.getGroundTruths().get(0));
		ArrayList<ArrayList<ArrayList<Double>>> dt = new ArrayList<ArrayList<ArrayList<Double>>>();
		ArrayList<ArrayList<Integer>> gtt = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> pool = new ArrayList<Integer>();
		for(int i=0;i<TESTNUM;i++){
			dt.add(new ArrayList<ArrayList<Double>>());
			gtt.add(new ArrayList<Integer>());
			pool.add(i);
		}
		int n = d1.size();
		Random rnd = new Random();
		for(int i=0;i<n;i++){
			int rNum = rnd.nextInt(pool.size());
			int idx = pool.get(rNum);
			dt.get(idx).add(d1.get(i));
			gtt.get(idx).add(gt1.get(i));
			if(dt.get(idx).size()>=(double)n/TESTNUM){
				pool.remove(rNum);
			}
		}
		for(int i=0;i<TESTNUM;i++){
			ArrayList<ArrayList<Double>> trainD = new ArrayList<ArrayList<Double>>();
			ArrayList<Integer> trainGt = new ArrayList<Integer>();
			for(int j=0;j<TESTNUM;j++){
				if(i!=j){
					trainD.addAll(dt.get(j));
					trainGt.addAll(gtt.get(j));					
				}
			}
			ins[i] = new CNNInstance(
					trainD,
					trainGt,
					dt.get(i),
					gtt.get(i),
					datasets);
//			if(i==TESTNUM-1){
//				ins[i].setHideIterMsg(true);
//			}
//			System.out.println("Instance " + i + " training...");
			trd[i] = new Thread(ins[i]);
			trd[i].start();
		}
		
		try {
			for(int i=0;i<TESTNUM;i++){
				trd[i].join();
//				System.out.println(i+" joined");
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		for(int i=0;i<TESTNUM;i++){
//			ins[i].test(
//					dt.get(i),
//					gtt.get(i));
			Recorder re = ins[i].getRecorder();
			
			writeJson(re);
		}
	}

	private static void doTrainTest() {
		// Read input data
		datasets.readData(trainFile);
		datasets.readData(testFile);
		
		datasets.standardizeData();
		
		CNNInstance ins = new CNNInstance(
				datasets.getDatasets().get(0),
				datasets.getGroundTruths().get(0),
				datasets.getDatasets().get(1),
				datasets.getGroundTruths().get(1),
				datasets);
		
		ins.run();
//		ins.test(datasets.getDatasets().get(1), datasets.getGroundTruths().get(1));

		writeJson(ins.getRecorder());
	}

	@SuppressWarnings("unchecked")
	private static void writeJson(Recorder re) {
		JSONObject job = new JSONObject();
		job.put("name", "Output");
		JSONArray jar = new JSONArray();
		ObjectWithValue[] ary = re.ary;
		for(int j=ary.length-1;j>ary.length-11;j--){
			int hidlayidx = (int)ary[j].o;
			JSONObject job2 = new JSONObject();
			job2.put("name", datasets.getHiddenNodeContent(hidlayidx).name.replaceAll("\"", ""));
			JSONArray jar2 = new JSONArray();
			for(int k=0; k<datasets.getNetwork().size();k++){
				int l = 0;
				if(datasets.getNetwork().get(k).contains(hidlayidx)){
					JSONObject job3 = new JSONObject();
					job3.put("name", datasets.getLabelsName(k));
					jar2.add(job3);
					
					l++;
				}
				if(l>10) break;
			}
			job2.put("children", jar2);
			jar.add(job2);
		}
		job.put("children", jar);
		
		try {
			BufferedWriter bw;
			bw = new BufferedWriter(new FileWriter(new File("/home/cs682/public_html/result/"+ fileName +"_network.json")));
			bw.write(job.toJSONString());
			bw.close();
			bw = new BufferedWriter(new FileWriter(new File("/home/cs682/public_html/result/"+ fileName +"_ROC.json")));
			bw.write(re.AUCOutput());
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}
