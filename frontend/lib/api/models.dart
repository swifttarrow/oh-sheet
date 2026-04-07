/// JSON models matching backend/contracts.py and backend/jobs/events.py.
///
/// Kept deliberately small — only the fields the UI actually reads. The full
/// pydantic schemas live on the server; we mirror just enough to drive the
/// upload → progress → result flow.
class RemoteAudioFile {
  final String uri;
  final String format;
  final int sampleRate;
  final double durationSec;
  final int channels;
  final String? contentHash;

  RemoteAudioFile({
    required this.uri,
    required this.format,
    required this.sampleRate,
    required this.durationSec,
    required this.channels,
    this.contentHash,
  });

  factory RemoteAudioFile.fromJson(Map<String, dynamic> json) => RemoteAudioFile(
        uri: json['uri'] as String,
        format: json['format'] as String,
        sampleRate: json['sample_rate'] as int,
        durationSec: (json['duration_sec'] as num).toDouble(),
        channels: json['channels'] as int,
        contentHash: json['content_hash'] as String?,
      );

  Map<String, dynamic> toJson() => {
        'uri': uri,
        'format': format,
        'sample_rate': sampleRate,
        'duration_sec': durationSec,
        'channels': channels,
        if (contentHash != null) 'content_hash': contentHash,
      };
}

class RemoteMidiFile {
  final String uri;
  final int ticksPerBeat;
  final String? contentHash;

  RemoteMidiFile({
    required this.uri,
    required this.ticksPerBeat,
    this.contentHash,
  });

  factory RemoteMidiFile.fromJson(Map<String, dynamic> json) => RemoteMidiFile(
        uri: json['uri'] as String,
        ticksPerBeat: json['ticks_per_beat'] as int,
        contentHash: json['content_hash'] as String?,
      );

  Map<String, dynamic> toJson() => {
        'uri': uri,
        'ticks_per_beat': ticksPerBeat,
        if (contentHash != null) 'content_hash': contentHash,
      };
}

class JobSummary {
  final String jobId;
  final String status; // pending | running | succeeded | failed | cancelled
  final String variant;
  final String? title;
  final String? artist;
  final String? error;
  final Map<String, dynamic>? result;

  JobSummary({
    required this.jobId,
    required this.status,
    required this.variant,
    this.title,
    this.artist,
    this.error,
    this.result,
  });

  bool get isTerminal => status == 'succeeded' || status == 'failed' || status == 'cancelled';
  bool get succeeded => status == 'succeeded';

  factory JobSummary.fromJson(Map<String, dynamic> json) => JobSummary(
        jobId: json['job_id'] as String,
        status: json['status'] as String,
        variant: json['variant'] as String,
        title: json['title'] as String?,
        artist: json['artist'] as String?,
        error: json['error'] as String?,
        result: json['result'] as Map<String, dynamic>?,
      );
}

class JobEvent {
  final String jobId;
  final String type;
  final String? stage;
  final String? message;
  final double? progress;
  final Map<String, dynamic>? data;
  final String timestamp;

  JobEvent({
    required this.jobId,
    required this.type,
    this.stage,
    this.message,
    this.progress,
    this.data,
    required this.timestamp,
  });

  bool get isTerminal => type == 'job_succeeded' || type == 'job_failed';

  factory JobEvent.fromJson(Map<String, dynamic> json) => JobEvent(
        jobId: json['job_id'] as String,
        type: json['type'] as String,
        stage: json['stage'] as String?,
        message: json['message'] as String?,
        progress: (json['progress'] as num?)?.toDouble(),
        data: json['data'] as Map<String, dynamic>?,
        timestamp: json['timestamp'] as String,
      );
}

/// Stage execution order, used to estimate progress when the server doesn't
/// supply a numeric ``progress`` field on stage events. Mirrors
/// PipelineConfig.get_execution_plan() in the backend.
const List<String> kPipelineStages = [
  'ingest',
  'transcribe',
  'arrange',
  'humanize',
  'engrave',
];
