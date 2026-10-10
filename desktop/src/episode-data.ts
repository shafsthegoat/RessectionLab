import type {CasePayload,ResectionApi,ViewerCase} from './types.ts';
import type {DevelopmentEpisode,DevelopmentEpisodeResult,EpisodeOrigin,EpisodeAuthorship} from './episode-types.ts';
import type {ViewerReplay} from './viewer/contracts.ts';
import {checkedEpisodeAuthority} from './episode-authority.ts';
import {hydrateNativeEpisodeReplay} from './native-episode-replay.ts';
export interface EpisodeView {origin:EpisodeOrigin;authorship:EpisodeAuthorship|null;episode:DevelopmentEpisode;source:CasePayload;volume:ViewerCase;frames:ViewerReplay[];contactCounts:number[];probeCounts:number[]}
export async function hydrateDevelopmentEpisode(input:DevelopmentEpisodeResult,api:ResectionApi,origin:EpisodeOrigin='live'):Promise<EpisodeView>{
 const episode=structuredClone(input.episode);
 const authorship=checkedEpisodeAuthority(episode,structuredClone(input.episodeAuthorship),origin);
 if(episode.schema!=='resectionlab.shared-native-development-episode.v1')throw Error('Episode replay withheld: unsupported aspiration episode schema.');
 return {origin,authorship,...await hydrateNativeEpisodeReplay({...input,episode},api,6)};
}
