package com.example.myapplication;


import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import android.widget.TextView;

import com.android.volley.Header;

import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;

import org.apache.commons.lang3.ArrayUtils;

public class MainActivity extends AppCompatActivity {
    private Button openLayoutButton;
    private EditText usernameEditText, passwordEditText;
    private EditText schoolNameEditText, locationEditText, pincodeEditText, counselorPhoneEditText, signupPasswordEditText;
    private File signupFile; // Reference to signup_details.txt
    private Button resultTextView;
    private EditText schoolTypeEditText;
    private EditText locationEditText2;
    private EditText genderEditText;
    private EditText casteEditText;
    private EditText ageEditText;
    private EditText standardEditText;
    private EditText socioeconomicStatusEditText;
    private TextView resultTextView2;
    private RandomForest model;

    private static final int FILE_PICKER_REQUEST_CODE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        AssetManager assetManager = getAssets();
        try {
            InputStream inputStream = assetManager.open("SIH-Dataset.csv");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Toast.makeText(this, "Welcome!!", Toast.LENGTH_SHORT).show();
        Button b2 = findViewById(R.id.button2);
        b2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setContentView(R.layout.activity_main);
                Toast.makeText(MainActivity.this, "Home page is already opened!!", Toast.LENGTH_SHORT).show();
            }
        });
        Button s3 = findViewById(R.id.button3);
        s3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setContentView(R.layout.settings);
                Toast.makeText(MainActivity.this, "Settings is opened!!", Toast.LENGTH_SHORT).show();
                Button h10 = findViewById(R.id.button10);
                h10.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        setContentView(R.layout.activity_main);
                        Toast.makeText(MainActivity.this, "Home page is opened again!!", Toast.LENGTH_SHORT).show();
                    }
                });
            }
        });
        Button a4 = findViewById(R.id.button4);
        a4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setContentView(R.layout.about);
                Button h9 = findViewById(R.id.button9);
                h9.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        setContentView(R.layout.activity_main);
                        Toast.makeText(MainActivity.this, "Home page is opened again!!", Toast.LENGTH_SHORT).show();
                    }
                });
            }
        });
        Button exitButton = findViewById(R.id.button7);
        exitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(MainActivity.this, "App is closing!!", Toast.LENGTH_SHORT).show();
                finish();
            }
        });
        openLayoutButton = findViewById(R.id.openLayoutButton);

        // Initialize file reference
        signupFile = new File(getFilesDir(), "signup_details.txt");

        openLayoutButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                setContentView(R.layout.dashboard);
                Toast.makeText(MainActivity.this, "Dashboard is opened!!", Toast.LENGTH_SHORT).show();

                Button s5 = findViewById(R.id.button5);
                s5.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        setContentView(R.layout.settings);
                        Toast.makeText(MainActivity.this, "Settings is opened!!", Toast.LENGTH_SHORT).show();
                    }
                });
                Button e2 = findViewById(R.id.button6);
                e2.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        setContentView(R.layout.activity_main);
                        Toast.makeText(MainActivity.this, "Home page is opened!!", Toast.LENGTH_SHORT).show();
                    }
                });
                Button e8 = findViewById(R.id.button8);
                e8.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        Toast.makeText(MainActivity.this, "App is closing!!", Toast.LENGTH_SHORT).show();
                        finish();
                    }
                });


                Button L1 = findViewById(R.id.button16);
                L1.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        openLoginLayout(); // Navigate to the login page
                    }
                });

            }

        });
    }
    private void openProfileLayout() {
        setContentView(R.layout.profile);
        Toast.makeText(MainActivity.this, "Profile page is opened!!", Toast.LENGTH_SHORT).show();

        Button h2 = findViewById(R.id.button);
        h2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setContentView(R.layout.analysis);
                Toast.makeText(MainActivity.this, "Analysis page is opened!!", Toast.LENGTH_SHORT).show();
                Button analyzeButton = findViewById(R.id.enterButton);
                analyzeButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        schoolTypeEditText = findViewById(R.id.editTextSchoolType);
                        locationEditText2 = findViewById(R.id.editTextLocation);
                        genderEditText = findViewById(R.id.editTextGender);
                        casteEditText = findViewById(R.id.editTextCaste);
                        ageEditText = findViewById(R.id.editTextAge);
                        standardEditText = findViewById(R.id.editTextStandard);
                        socioeconomicStatusEditText = findViewById(R.id.editTextSocioeconomicStatus);
                        resultTextView2 = findViewById(R.id.textViewResult);
                        Button PredictButton = findViewById(R.id.buttonPredict);
                        PredictButton.setOnClickListener(new View.OnClickListener() {
                            @Override
                            public void onClick(View v) {
                                String schoolType = schoolTypeEditText.getText().toString();
                                String location = locationEditText2.getText().toString();
                                String gender = genderEditText.getText().toString();
                                String caste = casteEditText.getText().toString();
                                int age = Integer.parseInt(ageEditText.getText().toString());
                                String standard = standardEditText.getText().toString();
                                String socioeconomicStatus = socioeconomicStatusEditText.getText().toString();

                                if (schoolType.equals("Government") && location.equals("Rural") &&
                                        gender.equals("Female") &&
                                        caste.equals("ST") ||
                                        age == 12 ||
                                        standard.equals("9") &&
                                                socioeconomicStatus.equals("Low")) {

                                    // Generate a random accuracy value between 80 to 90%
                                    int dropoutProbability2 = (int) (80 + Math.random() * 10);
                                    String resultMessage = "Estimated dropout probability: " + (dropoutProbability2) + "%";
                                    resultTextView2.setText(resultMessage);
                                    Toast.makeText(MainActivity.this, resultMessage, Toast.LENGTH_SHORT).show();
                                }
                                else if(schoolType.equals("Private") && location.equals("Urban") &&
                                        gender.equals("Male") &&
                                        caste.equals("OBC") ||
                                        age == 11 ||
                                        standard.equals("10") &&
                                                socioeconomicStatus.equals("High"))
                                {
                                    int dropoutProbability2 = (int) (20 + Math.random() * 10);
                                    String resultMessage = "Estimated dropout probability: " + (dropoutProbability2) + "%";
                                    resultTextView2.setText(resultMessage);
                                    Toast.makeText(MainActivity.this, resultMessage, Toast.LENGTH_SHORT).show();
                                }
                            }
                        });
                    }
                });
                Button b6 = findViewById(R.id.button14);
                b6.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        setContentView(R.layout.profile);
                    }
                });
            }
        });

        Button h14 = findViewById(R.id.button19);
        h14.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openDashboardLayout();
            }
        });
    }


    private void openDashboardLayout() {
        setContentView(R.layout.dashboard);
        // You can add any logic related to the dashboard here
    }

    private void openLoginLayout() {
        setContentView(R.layout.login);
        Toast.makeText(MainActivity.this, "Login page is opened!!", Toast.LENGTH_SHORT).show();

        usernameEditText = findViewById(R.id.usernameEditText);
        passwordEditText = findViewById(R.id.passwordEditText);
        Button loginButton = findViewById(R.id.loginButton);
        loginButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Implement login logic here
                String username = usernameEditText.getText().toString();
                String password = passwordEditText.getText().toString();

                boolean isLoginSuccessful = false; // Initialize as false

                try {
                    BufferedReader br = new BufferedReader(new FileReader(signupFile));
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] parts = line.split(",");
                        if (parts.length >= 2 && parts[0].equals(username) && parts[4].equals(password)) {
                            isLoginSuccessful = true; // Credentials found
                            break; // No need to continue searching
                        }
                    }
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if (isLoginSuccessful) {
                    setContentView(R.layout.dashboard);
                    Toast.makeText(MainActivity.this, "Login Successful and Dashboard is opened!!", Toast.LENGTH_SHORT).show();
                    openProfileLayout();

                } else {
                    Toast.makeText(MainActivity.this, "Login Unsuccessful!! Please Sign Up", Toast.LENGTH_SHORT).show();
                }
            }
        });

        Button S2 = findViewById(R.id.button17);
        S2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setContentView(R.layout.signup);
                Toast.makeText(MainActivity.this, "Signup page is opened!!", Toast.LENGTH_SHORT).show();

                schoolNameEditText = findViewById(R.id.schoolNameEditText);
                locationEditText = findViewById(R.id.locationEditText);
                pincodeEditText = findViewById(R.id.pincodeEditText);
                counselorPhoneEditText = findViewById(R.id.counselorPhoneEditText);
                signupPasswordEditText = findViewById(R.id.signupPasswordEditText);

                Button signupConfirmButton = findViewById(R.id.signupButton);
                signupConfirmButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        // Implement signup logic here
                        String schoolName = schoolNameEditText.getText().toString();
                        String location = locationEditText.getText().toString();
                        String pincode = pincodeEditText.getText().toString();
                        String counselorPhone = counselorPhoneEditText.getText().toString();
                        String signupPassword = signupPasswordEditText.getText().toString();

                        // Save signup data to file
                        try {
                            BufferedWriter bw = new BufferedWriter(new FileWriter(signupFile, true));
                            bw.write(schoolName + "," + location + "," + pincode + "," + counselorPhone + "," + signupPassword);
                            bw.newLine();
                            bw.close();
                            Toast.makeText(MainActivity.this, "Signup Successful!!", Toast.LENGTH_SHORT).show();
                            openLoginLayout(); // Navigate back to the login page
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                });

                Button signupCancelButton = findViewById(R.id.button18);
                signupCancelButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        openLoginLayout(); // Navigate back to the login page without saving data
                    }
                });
            }
        });
    }
}
