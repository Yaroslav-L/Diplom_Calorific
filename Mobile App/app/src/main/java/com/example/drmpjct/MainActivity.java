package com.example.drmpjct;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Point;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.text.format.Time;
import android.util.JsonReader;
import android.view.Display;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;
import androidx.recyclerview.widget.RecyclerView;

import com.example.drmpjct.databinding.ActivityMainBinding;
import com.google.android.material.bottomnavigation.BottomNavigationView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONStringer;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    static final int GALLERY_REQUEST = 1;
    static final int CAMERA_REQUEST = 2;

    String btmpnm;
    JSONObject jsonM;
    String[] arrayList;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
                (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.INTERNET) != PackageManager.PERMISSION_GRANTED) ||
                (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED)) {
            requestPermissions(new String[]{Manifest.permission.CAMERA,Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.INTERNET}, 1);
        }


        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        //Обьявление фрагментов, навигатора и их связь
        BottomNavigationView navView = findViewById(R.id.nav_view);
        AppBarConfiguration appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.navigation_plan, R.id.navigation_cam, R.id.navigation_profile)
                .build();
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_activity_main);
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);
        NavigationUI.setupWithNavController(binding.navView, navController);
        navController.navigate(R.id.navigation_cam);

    }

    //Обработка программно созданных страниц
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent imageReturnedIntent) {
        super.onActivityResult(requestCode, resultCode, imageReturnedIntent);

        Bitmap bitmap = null;
        ImageView imageView = (ImageView) findViewById(R.id.imgview_photo);

        //Определение страницы
        switch(requestCode) {
            case GALLERY_REQUEST:
                if(resultCode == RESULT_OK){
                   Uri selectedImage = imageReturnedIntent.getData();
                    try {
                        bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImage);
                    } catch (IOException e) {
                        e.printStackTrace();
                        break;
                    }
                    imageView.setImageBitmap(bitmap);
                }
                break;
            case CAMERA_REQUEST:
                if(resultCode == RESULT_OK) {
                    Bundle extras = imageReturnedIntent.getExtras();
                    bitmap = (Bitmap) extras.get("data");
                    imageView.setImageBitmap(bitmap);
                }break;
        }
    }

    //Оработчик кнопки галлерея
    public void btngalclc(View view){
        Toast.makeText(getApplicationContext(), "Выберите фото еды", Toast.LENGTH_LONG).show();
        //Создание и активация стрницы Галлерея
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        photoPickerIntent.setType("image/*");
        startActivityForResult(photoPickerIntent, GALLERY_REQUEST);
    }

    //Оработчик кнопки Камера
    public void btncamclc(View view) throws IOException {
        Toast.makeText(getApplicationContext(), "Сделайте фото еды", Toast.LENGTH_LONG).show();
        //Создание и активация стрницы камеры
        Intent camIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(camIntent, CAMERA_REQUEST);
    }

    //Оработчик кнопки Анализ
    public void btnanalclc(View view) throws IOException, JSONException {
        Toast.makeText(getApplicationContext(), "Производится анализ", Toast.LENGTH_LONG).show();

        ImageView imageView = (ImageView) findViewById(R.id.imgview_photo);
        Bitmap bitmap = null;
        bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
        Time time = new Time();  time.setToNow();  btmpnm = String.valueOf(time);
        OutputStream os = null;
        try{
            os = openFileOutput("000001.jpeg", MODE_PRIVATE);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, os);
            os.flush();
            os.close();
        } catch (IOException e){e.printStackTrace();}

        File file = new File("/data/data/com.example.drmpjct/files/000001.jpeg");
        OkHttpClient client = new OkHttpClient().newBuilder()
                .connectTimeout(2, TimeUnit.MINUTES)
                .readTimeout(2, TimeUnit.MINUTES)
                .writeTimeout(2, TimeUnit.MINUTES)
                .build();
        MediaType mediaType = MediaType.parse("multipart/form-data");
        RequestBody body = new MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("image","000001.jpeg.jpeg",
            RequestBody.create(MediaType.parse("application/octet-stream"),file)).build();

        Request request;
        RadioButton rb1 = findViewById(R.id.radioButton);
        RadioButton rb2 = findViewById(R.id.radioButton2);

        if (rb1.isChecked()){
            request = new Request.Builder()
                    .url("http://192.168.0.40:8000/classif")
                    .method("POST", body)
                    .addHeader("Content-Type", "multipart/form-data")
                    .addHeader("Accept", "application/json")
                    .build();
        }
        else {
            request = new Request.Builder()
            .url("http://192.168.0.40:8000/detect")
                    .method("POST", body)
                    .addHeader("Content-Type", "multipart/form-data")
                    .addHeader("Accept", "application/json")
                    .build();
        }

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                Toast.makeText(getApplicationContext(), "Анализ не удался. Пожалуйста, повторите попытку.", Toast.LENGTH_LONG).show();
            }
            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {

                String resStr = response.body().string();

                JSONObject jsonn;
                String n = "";
                String k = "";

                try {
                    jsonn = new JSONObject(resStr);
                    n = (String) jsonn.getString("name");
                    k = (String) jsonn.getString("KcaL");
                } catch (JSONException e) {
                    //Toast.makeText(getApplicationContext(), "Анализ не удался. Пожалуйста, повторите попытку.", Toast.LENGTH_LONG).show();
                }

                String finalK = k;
                String finalN = n;
                System.out.print(finalK+" "+finalN);
                runOnUiThread(new Runnable() {

                    @Override
                    public void run() {

                        TextView txtview = findViewById(R.id.txt_cal_result2);
                        TextView txtview2 = findViewById(R.id.txt_cal2);
                        txtview.setText(finalK+" кКал");
                        txtview2.setText(finalN);

                    }
                });
            }
        });
    }
}

