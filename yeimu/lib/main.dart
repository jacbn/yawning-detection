import 'dart:async';

import 'package:flutter/material.dart';
import 'package:esense_flutter/esense.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:yeimu/results.dart';
import 'package:yeimu/structure/sensor_reading.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Yeimu',
      theme: ThemeData(
        primarySwatch: Colors.purple,
      ),
      home: const MainPage(title: 'Yeimu'),
    );
  }
}

class MainPage extends StatefulWidget {
  const MainPage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  String _eSenseId = 'eSense-0662';
  late ESenseManager eSenseManager;
  late StreamSubscription subscription;

  bool _isCollecting = false;
  bool _isConnected = false;
  Icon _recordIcon = const Icon(Icons.play_arrow);
  int _sampleRate = 32;
  double _sliderVal = 4.0;
  String _eventsText = '';
  String _connectButtonText = 'Connect';
  List<SensorReading> _sensorEvents = [];

  _MainPageState() {
    eSenseManager = ESenseManager(_eSenseId);
    eSenseManager.setSamplingRate(_sampleRate);
  }

  Future<bool> _connect() async {
    _printEvent("Attempting to connect to $_eSenseId");
    if (await _hasPermissions()) {
      _printEvent("Permissions granted. Connecting...");

      // add a listener to the connection event stream, to allow events to be printed to the app
      StreamSubscription connectionEvents = eSenseManager.connectionEvents.listen((event) {
        _printEvent('Connection event: $event');
      });
      _printEvent("Connection event listener added");

      // eSense BTLE interface does not work with fast, consecutive calls; add a delay
      await Future.delayed(const Duration(seconds: 1));

      // initiate the connection process
      bool connecting = await eSenseManager.connect();
      _printEvent("Connection scanning started: $connecting");

      // if an error with initiation occurred, stop
      if (!connecting) {
        return false;
      }

      // give some time for the connection to be established
      // TODO: await on the actual connection event instead of a timer
      await Future.delayed(const Duration(seconds: 5));

      // check if the connection was successful
      bool connected = await eSenseManager.isConnected();
      _printEvent("Connection complete: $connected\n");

      if (connected) {
        setState(() {
          _isConnected = true;
          _connectButtonText = 'Disconnect';
        });
      }

      return connected;
    } else {
      _newAlert("No permissions", "Lacking permissions. Please grant the requested permissions.");
      _printEvent("No permissions.");
      return false;
    }
  }

  Future<bool> _disconnect() async {
    _printEvent("Disconnecting...");
    bool disconnected = await eSenseManager.disconnect();
    if (disconnected) {
      setState(() {
        _connectButtonText = 'Connect';
        _isConnected = false;
      });
    }
    return disconnected;
  }

  Future<bool> _hasPermissions() async {
    Map<Permission, PermissionStatus> statuses = await [
      Permission.bluetooth,
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
      Permission.location, // precise location is needed to find nearby bluetooth devices
    ].request();
    return !statuses.values.map((e) => e == PermissionStatus.granted).contains(false);
  }

  void _printEvent(String event) {
    setState(() {
      if (_eventsText == '') {
        _eventsText = event;
      } else {
        _eventsText += '\n$event';
      }
    });
  }

  void _toggleDataCollection() async {
    if (!eSenseManager.connected) {
      bool connect = await _connect();
      if (!connect) {
        _newAlert(
          "Not Connected",
          "No eSense device connected. Connect via Bluetooth and try again.",
        );
      }
      return;
    }

    setState(() {
      _isCollecting = !_isCollecting;
      if (_isCollecting) {
        eSenseManager.setSamplingRate(_sampleRate);
        _printEvent("Sampling rate set to $_sampleRate Hz");
        subscription = eSenseManager.sensorEvents.listen((event) {
          _printEvent('Sensor event: $event');
          _sensorEvents.add(SensorReading(event.accel, event.gyro));
        });
        _recordIcon = const Icon(Icons.save);
      } else {
        subscription.cancel();
        _recordIcon = const Icon(Icons.play_arrow);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: <Widget>[
            Padding(
              padding: const EdgeInsets.only(left: 100, right: 100, top: 20),
              child: TextFormField(
                initialValue: _eSenseId,
                decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Device Name',
                    suffixIcon: Icon(Icons.bluetooth),
                    isDense: true,
                    contentPadding: EdgeInsets.all(8.0)),
                onChanged: (text) {
                  setState(() {
                    _eSenseId = text;
                  });
                },
              ),
            ),
            TextButton(
              onPressed: () {
                _isConnected ? _disconnect() : _connect();
              },
              child: Text(_connectButtonText, style: const TextStyle(color: Colors.blue)),
            ),
            const SizedBox(height: 30),
            GridView.count(
              crossAxisCount: 2,
              crossAxisSpacing: 10,
              mainAxisSpacing: 30,
              childAspectRatio: 2.8,
              shrinkWrap: true,
              padding: const EdgeInsets.all(10),
              children: [
                SizedBox(
                  height: 100,
                  child: Column(
                    children: [
                      Text("Sampling Rate: $_sampleRate Hz"),
                      Slider(
                        max: 12.0,
                        min: 1.0,
                        value: _sliderVal,
                        divisions: 12,
                        label: "$_sampleRate Hz",
                        onChanged: (value) {
                          setState(() {
                            _sliderVal = value;
                            _sampleRate = value.toInt() * 8;
                          });
                        },
                      ),
                    ],
                  ),
                ),
                Center(
                  child: Text(
                    "Recording Data: \n\n$_isCollecting\n",
                    textAlign: TextAlign.center,
                  ),
                ),
                ElevatedButton(
                  style: ButtonStyle(
                      backgroundColor: MaterialStateProperty.all(Theme.of(context).primaryColor)),
                  onPressed: () {
                    Navigator.push(context,
                        MaterialPageRoute(builder: (context) => Results(data: _sensorEvents)));
                  },
                  child: const Text("View Results", style: TextStyle(color: Colors.white)),
                ),
                ElevatedButton(
                  style: ButtonStyle(
                      backgroundColor: MaterialStateProperty.all(Theme.of(context).primaryColor)),
                  onPressed: () {
                    setState(() {
                      _sensorEvents.clear();
                      _eventsText = '';
                    });
                  },
                  child: const Text("Clear Log", style: TextStyle(color: Colors.white)),
                ),
              ],
            ),
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(width: 1.0),
                ),
                child: SizedBox(
                  height: 150,
                  width: MediaQuery.of(context).size.width,
                  child: SingleChildScrollView(
                    child: Padding(
                      padding: const EdgeInsets.all(4.0),
                      child: Text(_eventsText),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _toggleDataCollection,
        tooltip: 'Toggle Data Collection',
        child: _recordIcon,
      ),
    );
  }

  void _newAlert(String title, String body) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(title),
          content: Text(body),
          actions: [
            TextButton(
              child: const Text("OK"),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }
}
