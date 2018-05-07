package com.aibd.watson;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.*;
import java.util.Base64;


import javax.net.ssl.HttpsURLConnection;

public class LanguageTranslator {
	
	public static void main(String[] args) throws Exception{
		
		//String httpsURL = "https://gateway.watsonplatform.net/language-translator/api/v2/translate?text=Hello&source=en&target=es";
		String httpsURL = "https://gateway.watsonplatform.net/language-translator/api/v2/translate";
		URL myurl = new URL(httpsURL);
		HttpsURLConnection connection = (HttpsURLConnection)myurl.openConnection();
		
		connection.setDoInput(true);
        connection.setDoOutput(true);
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Accept", "text/plain");
        
        connection.setRequestProperty("model_id", "en-fr");
        connection.setRequestProperty("text", "hello");
        
		
		String userpass = "83ba4375-35b7-43ca-897a-90032efb96d5" + ":" + "BzXJeCKGu3a0";
		String basicAuth = "Basic " + javax.xml.bind.DatatypeConverter.printBase64Binary(userpass.getBytes());
		
		connection.setRequestProperty ("Authorization", basicAuth);
		

		
		
		
	
		BufferedReader br = new BufferedReader(new InputStreamReader(connection.getInputStream()));
        StringBuffer jsonString = new StringBuffer();
        String line;
        while ((line = br.readLine()) != null) {
                jsonString.append(line);
        }
        System.out.println(jsonString);
        br.close();
        connection.disconnect();
	}

}
