import 'dart:async';

import 'package:flutter/material.dart';
import 'package:esense_flutter/esense.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:yeimu/io/io_manager.dart';
import 'package:yeimu/results.dart';
import 'package:yeimu/structure/sensor_reading.dart';
import 'package:yeimu/structure/timestamps.dart';

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
  String _recordingName = '';
  TimestampType _timestampType = TimestampType.yawn;
  late ESenseManager eSenseManager;
  late StreamSubscription connectionEvents;
  late StreamSubscription sensorEvents;

  bool _isCollecting = false;
  bool _isConnected = false;
  Icon _recordIcon = const Icon(Icons.play_arrow);
  int _sampleRate = 32;
  double _sliderVal = 4.0;
  String _eventsText = '';
  String _connectButtonText = 'Connect';
  final List<SensorReading> _sensorReadings = [];
  final List<Timestamp> _timestamps = [];

  _MainPageState() {
    eSenseManager = ESenseManager(_eSenseId);
    eSenseManager.setSamplingRate(_sampleRate);
  }

  void _initiateConnection() async {
    bool connecting = false;
    _printEvent("Attempting to connect to $_eSenseId");
    if (await _hasPermissions()) {
      _printEvent("Permissions granted. Connecting...");

      // add a listener to the connection event stream to allow events to be printed to the app
      connectionEvents = eSenseManager.connectionEvents.listen((event) {
        _printEvent('Connection event: $event');
        // once a 'connected' event is found (and we were expecting it), update the app state to reflect this
        if (connecting && event.type == ConnectionType.connected) {
          _completeConnection();
        }
      });
      _printEvent("Connection event listener added");

      // eSense BTLE interface does not work with fast, consecutive calls; add a delay
      await Future.delayed(const Duration(seconds: 1));

      // initiate the connection process
      connecting = await eSenseManager.connect();
      _printEvent("Connection scanning started: $connecting");

      // await _completeConnection() call from the connection events listener
    } else {
      _newAlert("No permissions", "Lacking permissions. Please grant the requested permissions.");
      _printEvent("No permissions.");
    }
  }

  void _completeConnection() async {
    // check eSense device sees connection as successful
    bool connected = await eSenseManager.isConnected();
    _printEvent("Connection complete: $connected\n");

    // update app state
    if (connected) {
      setState(() {
        _isConnected = true;
        _connectButtonText = 'Disconnect';
      });
    }

    // stop receiving events on this listener as it is now redundant (even if the connection fails, as we'd make a new one when trying again)
    connectionEvents.cancel();
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
      _newAlert(
        "Not Connected",
        "No eSense device connected. Connect via Bluetooth and try again.",
      );
      return;
    }

    setState(() {
      _isCollecting = !_isCollecting;
    });
    if (_isCollecting) {
      setState(() {
        eSenseManager.setSamplingRate(_sampleRate);
        _printEvent("Sampling rate set to $_sampleRate Hz");

        // add new listener to sensor event stream, with a function to add any new reading to _sensorReadings
        sensorEvents = eSenseManager.sensorEvents.listen((event) {
          _printEvent('Sensor event: $event');
          _sensorReadings.add(SensorReading(event.accel ?? [], event.gyro ?? []));
        });

        _sensorReadings.clear();
        _timestamps.clear();
        _recordIcon = const Icon(Icons.save);
      });
    } else {
      sensorEvents.cancel();
      //must be outside of setState
      await IOManager.saveData(
          (_recordingName.isEmpty) ? null : _recordingName.trim(), _sensorReadings, _timestamps);
      setState(() {
        _recordIcon = const Icon(Icons.play_arrow);
        _sensorReadings.clear();
        _timestamps.clear();
      });
    }
  }

  void addTimestamp() {
    if (_isCollecting) {
      setState(() {
        _timestamps.add(Timestamp(_sensorReadings.length, _timestampType));
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: <Widget>[
            // device name input field
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
            // connect button
            TextButton(
              onPressed: () {
                _isConnected ? _disconnect() : _initiateConnection();
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
              physics: const NeverScrollableScrollPhysics(),
              padding: const EdgeInsets.all(10),
              children: [
                SizedBox(
                  height: 100,
                  child: Column(
                    children: [
                      // sampling rate
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
                // is recording data
                Center(
                  child: Text(
                    "Recording Data: \n\n$_isCollecting\n",
                    textAlign: TextAlign.center,
                  ),
                ),
                // results button
                ElevatedButton(
                  style: ButtonStyle(
                      backgroundColor: MaterialStateProperty.all(Theme.of(context).primaryColor)),
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => Results(
                          data: _sensorReadings,
                          timestamps: _timestamps,
                        ),
                      ),
                    );
                  },
                  child: const Text("View Results", style: TextStyle(color: Colors.white)),
                ),
                // clear log button
                ElevatedButton(
                  style: ButtonStyle(
                      backgroundColor: MaterialStateProperty.all(Theme.of(context).primaryColor)),
                  onPressed: () {
                    setState(() {
                      _sensorReadings.clear();
                      _timestamps.clear();
                      _eventsText = '';
                    });
                  },
                  child: const Text("Clear Log", style: TextStyle(color: Colors.white)),
                ),
              ],
            ),
            // log
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
            const Text("Timestamp:"),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // timestamp dropdown menu
                DropdownButton<String>(
                  value: _timestampType.string,
                  items: TimestampType.values
                      .map((type) => DropdownMenuItem<String>(
                            value: type.string,
                            // title case the string for display
                            child: Text(type.string
                                .replaceRange(0, 1, type.string.substring(0, 1).toUpperCase())),
                          ))
                      .toList(),
                  onChanged: (value) => setState(() {
                    _timestampType = Timestamp.eventFromString(value ?? TimestampType.other.string);
                  }),
                ),
                const SizedBox(width: 10),
                // add timestamp button
                ElevatedButton(
                  style: ButtonStyle(
                      backgroundColor: MaterialStateProperty.all(Theme.of(context).primaryColor)),
                  onPressed: () {
                    addTimestamp();
                  },
                  child: const Text("Add", style: TextStyle(color: Colors.white)),
                ),
              ],
            ),
            const SizedBox(height: 30),
            Padding(
              padding: const EdgeInsets.only(left: 100, right: 100, top: 20),
              child: TextFormField(
                initialValue: _recordingName,
                decoration: const InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Recording Name',
                    isDense: true,
                    contentPadding: EdgeInsets.all(16.0)),
                onChanged: (text) {
                  setState(() {
                    _recordingName = text;
                  });
                },
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
