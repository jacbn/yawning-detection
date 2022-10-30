import 'dart:async';
import 'dart:math';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:yeimu/structure/sensor_reading.dart';
import 'package:collection/collection.dart';

class Results extends StatefulWidget {
  const Results({super.key, required this.data});
  final List<SensorReading> data;

  @override
  State<Results> createState() => _ResultsState();
}

class _ResultsState extends State<Results> {
  List<Color> axisColors = [Colors.red, Colors.green, Colors.blue];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Results'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text('Accelerometer Data', style: Theme.of(context).textTheme.headline5),
          Padding(
            padding: const EdgeInsets.only(left: 0, top: 8, right: 8, bottom: 2),
            child: AspectRatio(
              aspectRatio: 1.5,
              child: DecoratedBox(
                decoration: const BoxDecoration(
                  borderRadius: BorderRadius.all(Radius.circular(10)),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: LineChart(accelData(widget.data)),
                ),
              ),
            ),
          ),
          Text('Gyroscope Data', style: Theme.of(context).textTheme.headline5),
          Padding(
            padding: const EdgeInsets.only(left: 0, top: 8, right: 8, bottom: 2),
            child: AspectRatio(
              aspectRatio: 1.5,
              child: DecoratedBox(
                decoration: const BoxDecoration(
                  borderRadius: BorderRadius.all(Radius.circular(10)),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: LineChart(gyroData(widget.data)),
                ),
              ),
            ),
          ),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text("Legend:", style: Theme.of(context).textTheme.bodyLarge),
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(width: 20),
                  legendItem(Colors.red, 'X'),
                  const SizedBox(width: 20),
                  legendItem(Colors.green, 'Y'),
                  const SizedBox(width: 20),
                  legendItem(Colors.blue, 'Z'),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  Row legendItem(Color color, String textVal) {
    return Row(
      children: [
        Container(
          width: 20,
          height: 20,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10),
            color: color,
          ),
        ),
        const SizedBox(width: 5),
        Text(textVal),
      ],
    );
  }

  LineChartData gyroData(List<SensorReading> allData) {
    List<LineChartBarData> gyroAxisReadings = getIMUReadings(allData, (v) => v.gyro);
    return makeLineChartData(gyroAxisReadings, allData.length);
  }

  LineChartData accelData(List<SensorReading> allData) {
    List<LineChartBarData> gyroAxisReadings = getIMUReadings(allData, (v) => v.accel);
    return makeLineChartData(gyroAxisReadings, allData.length);
  }

  List<LineChartBarData> getIMUReadings(List<SensorReading> allData, Function sensorReadingToAxes) {
    List<LineChartBarData> gyroAxisReadings = axisColors
        .mapIndexed(
          (chartId, col) => LineChartBarData(
            spots: allData
                .mapIndexed(
                    (i, v) => FlSpot(i.toDouble(), sensorReadingToAxes(v)![chartId].toDouble()))
                .toList(),
            isCurved: true,
            isStrokeCapRound: true,
            color: col,
          ),
        )
        .toList();
    return gyroAxisReadings;
  }

  LineChartData makeLineChartData(List<LineChartBarData> data, int numPoints) {
    double minY = (data[0].spots.isEmpty) ? -1 : data.map((e) => e.mostBottomSpot.y).reduce(min);
    double maxY = (data[0].spots.isEmpty) ? 1 : data.map((e) => e.mostTopSpot.y).reduce(max);
    double horizontalInterval = (maxY - minY) / 8;
    double verticalInterval = max(1, numPoints / 8);
    return LineChartData(
      minX: 0,
      minY: minY,
      maxX: numPoints - 1.0,
      maxY: maxY,
      gridData: FlGridData(
        show: true,
        drawVerticalLine: true,
        horizontalInterval: horizontalInterval,
        verticalInterval: verticalInterval,
      ),
      lineBarsData: data,
      titlesData: FlTitlesData(
        show: true,
        rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        bottomTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 30,
            interval: verticalInterval,
          ),
        ),
        leftTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 50,
            interval: horizontalInterval,
          ),
        ),
      ),
    );
  }
}
