import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.trees.*;
import java.util.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;


public class SentimentAnnotator {

    public static void main(String[] args) throws IOException {
    
        // CoreNLP
        Properties props = new Properties();
        props.setProperty(
            "annotators",
            "tokenize,ssplit,pos,parse,sentiment");
        props.setProperty(
            "timeout",
            "30000");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Data folder
        File folder = new File("data/");
        File[] listOfFiles = folder.listFiles();

        // Process each file
        for (File file : listOfFiles) {
            if (file.isFile()) {

                // Skip
                if (!file.getName().endsWith(".txt")) continue;
                System.out.println(file.getName());

                // Read file                
                Scanner scan = new Scanner(new File(file.getAbsolutePath()));
                      
                // Output file path
                String fp = file.getAbsolutePath() + ".sentiment.txt";
                BufferedWriter writer = new BufferedWriter(new FileWriter(fp));

                int cnt = 0;

                while(scan.hasNextLine()){

                    cnt++;
                    if ((cnt % 50) == 0) System.out.println(cnt);

                    // Get text
                    String text = scan.nextLine();

                    // Annotate
                    Annotation annotation = pipeline.process(text);

                    int total = 0;
                    int sentence_cnt = 0;

                    for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {

                        sentence_cnt++;

                        Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
                        int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
                        total += sentiment;
                    }
                    // Get average
                    double avg = total / (double) sentence_cnt;
                    writer.write(String.valueOf(avg) + "\n");
                }
                writer.close();
            }
        }
    }
}

