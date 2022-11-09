enum TimestampType {
  yawn,
  other,
}

class Timestamp {
  int time;
  TimestampType type;

  Timestamp(this.time, this.type);
  Timestamp.fromEventName(this.time, String eventName) : type = eventFromString(eventName);

  @override
  String toString() {
    return '$time ${type.string}';
  }

  // can't add functions to an enum with extensions so this must be in the Timestamp class instead
  static TimestampType eventFromString(String s) {
    switch (s) {
      case 'yawn':
        return TimestampType.yawn;
      default:
        return TimestampType.other;
    }
  }
}

extension StringOps on TimestampType {
  String get string {
    switch (this) {
      case TimestampType.yawn:
        return 'yawn';
      case TimestampType.other:
        return 'other';
    }
  }
}
