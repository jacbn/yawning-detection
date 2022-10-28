import 'dart:async';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class Results extends StatelessWidget {
  const Results({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Results'),
      ),
      body: Padding(
        padding: const EdgeInsets.only(left: 0, top: 8, right: 8, bottom: 2),
        child: AspectRatio(
          aspectRatio: 1.5,
          child: DecoratedBox(
            decoration: const BoxDecoration(
              borderRadius: BorderRadius.all(Radius.circular(10)),
            ),
            child: Padding(
              padding: const EdgeInsets.all(8),
              child: LineChart(accelData()),
            ),
          ),
        ),
      ),
    );
  }

  LineChartData accelData() {
    return LineChartData(
      minX: 0,
      minY: -3600,
      maxX: 100,
      maxY: 3600,
      gridData: FlGridData(
        show: true,
        drawVerticalLine: true,
        horizontalInterval: 1200,
        verticalInterval: 10,
      ),
      lineBarsData: [
        LineChartBarData(
          spots: const [
            FlSpot(0, 1800),
            FlSpot(40, 0),
            FlSpot(80, -200),
          ],
          isCurved: true,
          isStrokeCapRound: true,
        ),
      ],
      titlesData: FlTitlesData(
          show: true,
          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              interval: 10,
            ),
          ),
          leftTitles: AxisTitles(
              sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 50,
            interval: 1200,
          ))),
    );
  }
}
