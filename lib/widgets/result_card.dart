import 'package:flutter/material.dart';

import '../services/scan_service.dart';
import '../strings.dart';

/// What a result means, in words and colour; used by the Result screen and History.
/// No treatment advice is shown anywhere.
class ResultCard extends StatelessWidget {
  const ResultCard({super.key, required this.result});
  final ScanResult result;

  @override
  Widget build(BuildContext context) {
    final r = result;
    switch (r.outcome) {
      case 'healthy':
        return const MessageCard(icon: Icons.check_circle, color: Color(0xFF2E7D32), title: S.healthyTitle, lines: [S.healthyBody]);
      case 'severity_only':
        return MessageCard(
          icon: Icons.warning_amber_rounded,
          color: const Color(0xFFE65100),
          title: S.damageTitle,
          lines: [S.severityLine(r.severity ?? '', r.damagePercent), S.noInsectSeen],
        );
      case 'pest':
        final (name, scientific) = S.pests[r.pest] ?? (r.pest ?? '', '');
        return MessageCard(
          icon: Icons.bug_report,
          color: const Color(0xFFC62828),
          title: name,
          subtitle: scientific,
          lines: [if (r.pestConfidence != null) S.howSure(r.pestConfidence!)], // the name only, no damage grade
        );
      case 'unknown_pest':
        return const MessageCard(
          icon: Icons.bug_report_outlined,
          color: Color(0xFF6D4C41),
          title: S.unknownPestTitle,
          lines: [S.unknownPestBody, S.unknownPestTip],
        );
      case 'not_corn':
        return MessageCard(icon: Icons.help_outline, color: Colors.grey.shade700, title: S.notCornTitle, lines: const [S.notCornBody]);
      default:
        return MessageCard(icon: Icons.blur_on, color: Colors.grey.shade700, title: S.unclearTitle, lines: const [S.unclearBody]);
    }
  }
}

class MessageCard extends StatelessWidget {
  const MessageCard({super.key, required this.icon, required this.color, required this.title, this.subtitle, this.lines = const []});
  final IconData icon;
  final Color color;
  final String title;
  final String? subtitle;
  final List<String> lines;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Icon(icon, size: 56, color: color),
            const SizedBox(height: 8),
            Text(title, textAlign: TextAlign.center, style: theme.textTheme.headlineSmall?.copyWith(color: color)),
            if (subtitle != null && subtitle!.isNotEmpty)
              Text(subtitle!, textAlign: TextAlign.center, style: const TextStyle(fontStyle: FontStyle.italic, fontSize: 16)),
            for (final line in lines) ...[
              const SizedBox(height: 10),
              Text(line, textAlign: TextAlign.center, style: theme.textTheme.bodyLarge),
            ],
          ],
        ),
      ),
    );
  }
}
